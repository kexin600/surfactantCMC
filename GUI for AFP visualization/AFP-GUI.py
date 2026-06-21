import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import AttentiveFP
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QSplitter,
    QMessageBox, QProgressBar, QFileDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QPainter, QColor
from PyQt5.QtWidgets import QStyle
from io import BytesIO

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationWorker(QThread):
    finished = pyqtSignal(object, object, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, smiles, device):
        super().__init__()
        self.model = model
        self.smiles = smiles
        self.device = device
    
    def get_atom_features(self, atom):
        atomic_num = atom.GetAtomicNum()
        common_atoms = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 35, 53, 80]
        atom_onehot = [0] * 20
        if atomic_num in common_atoms:
            atom_onehot[common_atoms.index(atomic_num)] = 1
        else:
            atom_onehot[-1] = 1
        
        degree = atom.GetDegree()
        degree_onehot = [0] * 5
        degree_onehot[min(degree, 4)] = 1
        
        formal_charge = float(atom.GetFormalCharge())
        
        chiral = atom.GetChiralTag()
        chiral_onehot = [0, 0, 0, 0]
        if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            chiral_onehot[0] = 1
        elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral_onehot[1] = 1
        else:
            chiral_onehot[3] = 1
        
        num_h = atom.GetTotalNumHs()
        num_h_norm = min(num_h, 4) / 4.0
        
        hybrid = atom.GetHybridization()
        hybrid_onehot = [0, 0, 0, 0]
        if hybrid == Chem.rdchem.HybridizationType.SP:
            hybrid_onehot[0] = 1
        elif hybrid == Chem.rdchem.HybridizationType.SP2:
            hybrid_onehot[1] = 1
        elif hybrid == Chem.rdchem.HybridizationType.SP3:
            hybrid_onehot[2] = 1
        else:
            hybrid_onehot[3] = 1
        
        is_aromatic = float(atom.GetIsAromatic())
        mass = atom.GetMass() / 200.0
        
        features = atom_onehot + degree_onehot + [formal_charge] + chiral_onehot + [num_h_norm] + hybrid_onehot + [is_aromatic] + [mass]
        if len(features) < 39:
            features += [0.0] * (39 - len(features))
        elif len(features) > 39:
            features = features[:39]
        return features
    
    def get_bond_features(self, bond):
        bond_type = bond.GetBondType()
        type_onehot = [0, 0, 0, 0]
        if bond_type == Chem.rdchem.BondType.SINGLE:
            type_onehot[0] = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            type_onehot[1] = 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            type_onehot[2] = 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            type_onehot[3] = 1
        
        stereo = bond.GetStereo()
        stereo_onehot = [0, 0, 0, 0]
        if stereo == Chem.rdchem.BondStereo.STEREONONE:
            stereo_onehot[0] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOANY:
            stereo_onehot[1] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOZ:
            stereo_onehot[2] = 1
        elif stereo == Chem.rdchem.BondStereo.STEREOE:
            stereo_onehot[3] = 1
        
        is_conjugated = float(bond.GetIsConjugated())
        is_in_ring = float(bond.IsInRing())
        
        features = type_onehot + stereo_onehot + [is_conjugated, is_in_ring]
        assert len(features) == 10, f"Edge feature dimension should be 10, got {len(features)}"
        return features
    
    def get_node_weights_afp(self, mol):
        atom_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float).to(self.device)
        
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            feat = self.get_bond_features(bond)
            edge_attrs.append(feat)
            edge_attrs.append(feat)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).to(self.device)
        
        first_conv_output = []
        def hook(module, input, output):
            first_conv_output.append(output)
        
        target_layer = None
        for name, module in self.model.named_modules():
            if 'convs.0' in name:
                target_layer = module
                break
            if 'conv' in name and isinstance(module, torch.nn.Module) and target_layer is None:
                target_layer = module
        
        if target_layer is None:
            print("Warning: No convolution layer found, returning uniform weights")
            return np.ones(mol.GetNumAtoms())
        
        handle = target_layer.register_forward_hook(hook)
        self.model.eval()
        with torch.no_grad():
            dummy_batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            _ = self.model(x, edge_index, edge_attr, dummy_batch)
        handle.remove()
        
        if first_conv_output:
            node_features = first_conv_output[0]
            node_weights = torch.norm(node_features, dim=1).cpu().numpy()
        else:
            print("Warning: No convolution output captured, returning uniform weights")
            node_weights = np.ones(x.size(0))
        
        if node_weights.max() - node_weights.min() > 1e-8:
            node_weights = (node_weights - node_weights.min()) / (node_weights.max() - node_weights.min())
        else:
            node_weights = np.ones_like(node_weights) * 0.5
        
        return node_weights
    
    def run(self):
        try:
            self.progress.emit(10)
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                self.error.emit("Invalid SMILES string. Please check your input!")
                return
            
            self.progress.emit(30)
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
            
            self.progress.emit(50)
            atom_weights = self.get_node_weights_afp(mol)
            self.progress.emit(80)
            self.finished.emit(mol, atom_weights, self.smiles)
            
        except Exception as e:
            self.error.emit(f"Visualization error: {str(e)}")


class AFPVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_ui()
        self.load_model()
    
    def create_emoji_icon(self, emoji):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setFont(QFont("Segoe UI Emoji", 48))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, emoji)
        painter.end()
        return QIcon(pixmap)
    
    def init_ui(self):
        self.setWindowTitle("Attentive FP Molecular Visualization System")
        self.setGeometry(200, 100, 1400, 900)
        
        style = self.style()
        self.setWindowIcon(self.create_emoji_icon("🧪"))
        
        global_font = QFont("Microsoft YaHei", 18)
        QApplication.setFont(global_font)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QGroupBox {
                font: bold 18px "Microsoft YaHei";
                border: 2px solid #d0d7de;
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #1f2937;
            }
            QLabel {
                font: 16px "Microsoft YaHei";
                padding: 8px;
                color: #374151;
            }
            QLineEdit {
                font: 16px "Microsoft YaHei";
                padding: 10px;
                border: 2px solid #d1d5db;
                border-radius: 8px;
                background-color: white;
                min-height: 45px;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                background-color: #fefce8;
            }
            QPushButton {
                font: bold 18px "Microsoft YaHei";
                padding: 10px 25px;
                border-radius: 8px;
                background-color: #3b82f6;
                color: white;
                border: none;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton#clearBtn {
                background-color: #ef4444;
            }
            QPushButton#clearBtn:hover {
                background-color: #dc2626;
            }
            QProgressBar {
                border: 2px solid #d1d5db;
                border-radius: 6px;
                text-align: center;
                background-color: #f3f4f6;
                font: 14px "Microsoft YaHei";
                min-height: 35px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 6px;
            }
            QSplitter::handle {
                background-color: #d1d5db;
                width: 3px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 15, 20, 20)
        
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setAlignment(Qt.AlignCenter)
        title_layout.setSpacing(15)
        
        logo_label = QLabel()
        self.load_company_logo(logo_label)
        title_layout.addWidget(logo_label)
        
        title_label = QLabel("Attentive FP Molecular Visualization System——TFRI")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Microsoft YaHei", 32, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("font-size: 32px; color: #1f2937; margin-bottom: 10px; padding: 0px;")
        title_layout.addWidget(title_label)
        
        main_layout.addWidget(title_widget)
        
        subtitle_label = QLabel("Molecular Feature Heatmap Visualization via Attention Mechanism")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont("Microsoft YaHei", 22)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("font-size: 22px; color: #6b7280; margin-bottom: 15px;")
        main_layout.addWidget(subtitle_label)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        smiles_group = QGroupBox("SMILES Input")
        smiles_layout = QVBoxLayout()
        smiles_layout.setSpacing(12)
        
        smiles_label = QLabel("SMILES String:")
        smiles_label.setFont(QFont("Microsoft YaHei", 16))
        smiles_layout.addWidget(smiles_label)
        
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("Enter SMILES string of the molecule")
        self.smiles_input.setMinimumHeight(50)
        self.smiles_input.setFont(QFont("Microsoft YaHei", 16))
        smiles_layout.addWidget(self.smiles_input)
        
        example_label = QLabel("Example Molecules:")
        example_label.setFont(QFont("Microsoft YaHei", 15))
        smiles_layout.addWidget(example_label)
        
        example_layout = QHBoxLayout()
        example_layout.setSpacing(12)
        
        self.example1_btn = QPushButton("Example 1")
        self.example1_btn.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.example1_btn.setStyleSheet("background-color: #8b5cf6; padding: 8px 12px; min-height: 45px;")
        self.example1_btn.clicked.connect(lambda: self.load_example("CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"))
        example_layout.addWidget(self.example1_btn)
        
        self.example2_btn = QPushButton("Example 2")
        self.example2_btn.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.example2_btn.setStyleSheet("background-color: #8b5cf6; padding: 8px 12px; min-height: 45px;")
        self.example2_btn.clicked.connect(lambda: self.load_example("CCCCCCCC[N+](C)(C)CCCC[N+](C)(C)CCCCCCCC"))
        example_layout.addWidget(self.example2_btn)
        
        self.example3_btn = QPushButton("Example 3")
        self.example3_btn.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.example3_btn.setStyleSheet("background-color: #8b5cf6; padding: 8px 12px; min-height: 45px;")
        self.example3_btn.clicked.connect(lambda: self.load_example("CCCCCCCC(CCCCCC)C1=CC=C(C=C1)S(=O)(=O)[O-].[Na+]"))
        example_layout.addWidget(self.example3_btn)
        
        example_layout.addStretch()
        smiles_layout.addLayout(example_layout)
        
        smiles_group.setLayout(smiles_layout)
        left_layout.addWidget(smiles_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(40)
        left_layout.addWidget(self.progress_bar)
        
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        self.visualize_btn = QPushButton("Start Visualization")
        self.visualize_btn.setMinimumHeight(60)
        self.visualize_btn.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.visualize_btn.clicked.connect(self.visualize_molecule)
        button_layout.addWidget(self.visualize_btn)
        
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setMinimumHeight(55)
        self.save_btn.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setMinimumHeight(55)
        self.clear_btn.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.clear_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_btn)
        
        button_group.setLayout(button_layout)
        left_layout.addWidget(button_group)
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        result_group = QGroupBox("Attention Heatmap")
        result_layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; padding: 10px; min-height: 600px;")
        self.image_label.setText("Waiting for SMILES input...\nClick 'Start Visualization' to generate attention heatmap")
        scroll_area.setWidget(self.image_label)
        result_layout.addWidget(scroll_area)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 850])
        
        main_layout.addWidget(splitter)
        
        self.statusBar().setFont(QFont("Microsoft YaHei", 13))
        self.statusBar().setMinimumHeight(35)
        self.statusBar().showMessage(f"Ready | Device: {self.device} | Waiting for model loading...")
    
    def load_model(self):
        try:
            model = AttentiveFP(
                in_channels=39,
                hidden_channels=128,
                out_channels=1,
                edge_dim=10,
                num_layers=3,
                num_timesteps=2,
                dropout=0.16928074796400608
            ).to(self.device)
            
            model_path = "early_stop_attentive_fp_best_model.pth"
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.model = model
            
            self.statusBar().showMessage(f"Model loaded successfully! Device: {self.device}", 3000)
            self.image_label.setText("Waiting for SMILES input...\nClick 'Start Visualization' to generate attention heatmap")
            
        except FileNotFoundError:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Model Loading Failed")
            msg_box.setText("Model file 'early_stop_attentive_fp_best_model.pth' not found.\nPlease ensure the model file is in the current directory.")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setFont(QFont("Microsoft YaHei", 14))
            msg_box.exec_()
            self.statusBar().showMessage("Model loading failed!", 3000)
            self.image_label.setText("Model loading failed!\nPlease check if the model file exists")
            
        except Exception as e:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"Model loading error: {str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setFont(QFont("Microsoft YaHei", 14))
            msg_box.exec_()
            self.statusBar().showMessage("Model loading error!", 3000)
            self.image_label.setText(f"Model loading error: {str(e)}")
    
    def load_example(self, smiles):
        self.smiles_input.setText(smiles)
        self.statusBar().showMessage(f"Loaded example SMILES: {smiles[:50]}...", 3000)
    
    def draw_molecule_matplotlib(self, mol, atom_weights, smiles):
        try:
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            
            pos = []
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                x, y = conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y
                pos.append((x, y))
            pos = np.array(pos)
            
            atom_weights = np.array(atom_weights)
            w_min, w_max = atom_weights.min(), atom_weights.max()
            if w_max - w_min > 1e-8:
                norm_weights = (atom_weights - w_min) / (w_max - w_min)
            else:
                norm_weights = np.ones_like(atom_weights) * 0.5
            
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_aspect('equal')
            ax.axis('off')
            
            bond_segments = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_segments.append([pos[i], pos[j]])
            
            if bond_segments:
                lc = LineCollection(bond_segments, colors='gray', linewidths=2.5, alpha=0.7)
                ax.add_collection(lc)
            
            cmap_obj = plt.colormaps['coolwarm']
            colors = cmap_obj(norm_weights)
            ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=600, edgecolors='black', linewidth=0.3, zorder=2)
            
            for i, atom in enumerate(mol.GetAtoms()):
                ax.text(pos[i, 0], pos[i, 1], atom.GetSymbol(), ha='center', va='center',
                       fontsize=14, fontweight='bold', zorder=3)
            
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            x_padding = max(0.5, (x_max - x_min) * 0.15)
            y_padding = max(0.5, (y_max - y_min) * 0.15)
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cbar.set_label("Attention Weight (Importance)", fontsize=12)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            
            title_text = f"Molecule: {smiles[:60]}{'...' if len(smiles) > 60 else ''}"
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            
            qimage = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            
            scaled_pixmap = pixmap.scaled(self.image_label.size().width() - 20,
                                          self.image_label.size().height() - 20,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            plt.close(fig)
            return True
            
        except Exception as e:
            print(f"Drawing failed: {e}")
            return False
    
    def visualize_molecule(self):
        if self.model is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Model Not Loaded")
            msg_box.setText("The model has not been loaded. Please check the model file!")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setFont(QFont("Microsoft YaHei", 14))
            msg_box.exec_()
            return
        
        smiles = self.smiles_input.text().strip()
        if not smiles:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Input Error")
            msg_box.setText("Please enter a SMILES string!")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setFont(QFont("Microsoft YaHei", 14))
            msg_box.exec_()
            return
        
        self.visualize_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = VisualizationWorker(self.model, smiles, self.device)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_visualization_complete)
        self.worker.error.connect(self.on_visualization_error)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def on_visualization_complete(self, mol, atom_weights, smiles):
        try:
            success = self.draw_molecule_matplotlib(mol, atom_weights, smiles)
            
            if success:
                self.statusBar().showMessage(f"Visualization complete! Number of atoms: {len(atom_weights)}", 3000)
                self.save_btn.setEnabled(True)
            else:
                self.statusBar().showMessage("Visualization failed!", 3000)
                
        except Exception as e:
            self.statusBar().showMessage(f"Display failed: {str(e)}", 3000)
        finally:
            self.visualize_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def on_visualization_error(self, error_msg):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Visualization Error")
        msg_box.setText(error_msg)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setFont(QFont("Microsoft YaHei", 14))
        msg_box.exec_()
        
        self.image_label.setText(f"Visualization failed\n{error_msg}")
        self.visualize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def save_image(self):
        if not self.image_label.pixmap():
            QMessageBox.warning(self, "Warning", "No image to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG files (*.png);;JPEG files (*.jpg);;All files (*)"
        )
        
        if file_path:
            self.image_label.pixmap().save(file_path)
            self.statusBar().showMessage(f"Image saved: {file_path}", 3000)
    
    def clear_all(self):
        self.smiles_input.clear()
        self.image_label.clear()
        self.image_label.setText("Waiting for SMILES input...\nClick 'Start Visualization' to generate attention heatmap")
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("All content cleared", 2000)
    
    def closeEvent(self, event):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Exit")
        msg_box.setText("Are you sure you want to exit?")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setFont(QFont("Microsoft YaHei", 14))
        
        reply = msg_box.exec_()
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    def load_company_logo(self, logo_label):
        logo_paths = ["logo.png", "company_logo.png", "icon.png", "logo.jpg"]
        
        for path in logo_paths:
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    logo_label.setPixmap(scaled_pixmap)
                    logo_label.setStyleSheet("padding: 5px;")
                    return
            except:
                continue


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = AFPVisualizerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()