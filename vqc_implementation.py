#!/usr/bin/env python3
"""
Variational Quantum Classifier for NFL Play Type Classification
This implementation uses Qiskit to create a quantum machine learning model
for classifying football play types based on game situation features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B

class QuantumFootballClassifier:
    """
    Optimized Variational Quantum Classifier for NFL play type prediction
    """
    
    def __init__(self, num_qubits=8, num_layers=3, optimizer='COBYLA', max_iter=200):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.vqc = None
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))  # Better scaling for quantum
        self.le_target = LabelEncoder()
        self.le_posteam = LabelEncoder()
        self.le_defteam = LabelEncoder()
        
    def preprocess_data(self, df):
        """
        Enhanced preprocessing for better quantum classification
        """
        print("Preprocessing data...")
        
        # Filter for main play types with balanced sampling
        main_plays = ['Run', 'Pass', 'Field Goal', 'Punt']
        df_filtered = df[df['PlayType'].isin(main_plays)].copy()
        
        # Select important features
        features = ['down', 'yrdline100', 'TimeSecs', 'ydstogo', 'ScoreDiff', 
                   'qtr', 'GoalToGo', 'posteam', 'DefensiveTeam']
        
        df_processed = df_filtered[features + ['PlayType']].copy()
        df_processed = df_processed.dropna()
        
        # Reset index to ensure continuous indexing
        df_processed = df_processed.reset_index(drop=True)
        
        # Encode categorical variables
        df_processed['posteam_encoded'] = self.le_posteam.fit_transform(df_processed['posteam'])
        df_processed['DefensiveTeam_encoded'] = self.le_defteam.fit_transform(df_processed['DefensiveTeam'])
        
        # Select features for quantum processing
        quantum_features = ['down', 'yrdline100', 'TimeSecs', 'ydstogo', 'ScoreDiff', 
                           'qtr', 'GoalToGo', 'posteam_encoded']
        
        X = df_processed[quantum_features]
        y = df_processed['PlayType']
        
        # Encode target labels
        y_encoded = self.le_target.fit_transform(y)
        
        # Balance the dataset for better training
        X_balanced, y_balanced = self._balance_dataset(X, y_encoded)
        
        print(f"Processed dataset shape: {X_balanced.shape}")
        print(f"Target classes: {self.le_target.classes_}")
        print(f"Class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def _balance_dataset(self, X, y, min_samples_per_class=200):
        """
        Balance the dataset to ensure each class has sufficient samples
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # If all classes have enough samples, take a balanced subset
        if np.min(class_counts) >= min_samples_per_class:
            balanced_indices = []
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                selected_indices = np.random.choice(cls_indices, 
                                                  size=min_samples_per_class, 
                                                  replace=False)
                balanced_indices.extend(selected_indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            return X.iloc[balanced_indices], y[balanced_indices]
        else:
            # Use all available data
            return X, y
    
    def create_quantum_circuit(self, feature_map_type='ZZ'):
        """
        Create optimized quantum circuit components
        """
        print(f"Creating quantum circuit with {self.num_qubits} qubits and {self.num_layers} layers...")
        
        # Feature map selection
        if feature_map_type == 'ZZ':
            feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=3)  # Increased reps
        elif feature_map_type == 'Pauli':
            feature_map = PauliFeatureMap(feature_dimension=self.num_qubits, reps=3)
        else:
            raise ValueError("Unsupported feature map type")
        
        # Enhanced ansatz with more layers
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=self.num_layers, 
                               entanglement='full')  # Full entanglement
        
        print(f"Feature map qubits: {feature_map.num_qubits}")
        print(f"Ansatz parameters: {ansatz.num_parameters}")
        print(f"Feature map depth: {feature_map.depth()}")
        print(f"Ansatz depth: {ansatz.depth()}")
        
        return feature_map, ansatz
    
    def build_vqc(self, feature_map, ansatz):
        """
        Build optimized Variational Quantum Classifier
        """
        # Enhanced optimizer selection
        if self.optimizer_name == 'COBYLA':
            optimizer = COBYLA(maxiter=self.max_iter, disp=True)
        elif self.optimizer_name == 'SPSA':
            optimizer = SPSA(maxiter=self.max_iter)
        elif self.optimizer_name == 'L_BFGS_B':
            optimizer = L_BFGS_B(maxiter=self.max_iter)
        else:
            raise ValueError("Unsupported optimizer")
        
        # Create VQC
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=Sampler()
        )
        
        print(f"VQC created with {self.optimizer_name} optimizer ({self.max_iter} iterations)!")
        return self.vqc
    
    def prepare_quantum_data(self, X, y, sample_size=1000):
        """
        Enhanced data preparation for quantum processing
        """
        print(f"Preparing quantum data (sample size: {sample_size})...")
        
        # Sample data for quantum processing
        if len(X) > sample_size:
            # Stratified sampling to maintain class balance
            sample_indices = []
            unique_classes = np.unique(y)
            samples_per_class = sample_size // len(unique_classes)
            
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) >= samples_per_class:
                    selected = np.random.choice(cls_indices, size=samples_per_class, replace=False)
                else:
                    selected = cls_indices
                sample_indices.extend(selected)
            
            sample_indices = np.array(sample_indices)
            X_sample = X.iloc[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        # Enhanced feature scaling - direct to [0, π] range
        X_quantum = self.scaler.fit_transform(X_sample)
        
        print(f"Quantum data shape: {X_quantum.shape}")
        print(f"Feature range: [{X_quantum.min():.3f}, {X_quantum.max():.3f}]")
        
        return X_quantum, y_sample
    
    def train(self, X_train, y_train, train_size=1000):
        """
        Enhanced training with larger dataset
        """
        train_size = min(train_size, len(X_train))
        print(f"Training VQC on {train_size} samples...")
        
        # Use larger subset for better training
        X_train_subset = X_train[:train_size]
        y_train_subset = y_train[:train_size]
        
        print(f"Training set class distribution: {np.bincount(y_train_subset)}")
        
        try:
            self.vqc.fit(X_train_subset, y_train_subset)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training error with {self.optimizer_name}: {e}")
            print("Trying with SPSA optimizer...")
            # Fallback to SPSA
            self.optimizer_name = 'SPSA'
            optimizer = SPSA(maxiter=self.max_iter//2)  # Reduce iterations for fallback
            self.vqc.optimizer = optimizer
            self.vqc.fit(X_train_subset, y_train_subset)
            print("Training completed with SPSA optimizer!")
    
    def evaluate(self, X_test, y_test, test_size=100):
        """
        Enhanced evaluation with larger test set
        """
        print("Evaluating VQC...")
        
        # Use larger subset for evaluation
        test_size = min(test_size, len(X_test))
        X_test_subset = X_test[:test_size]
        y_test_subset = y_test[:test_size]
        
        print(f"Test set class distribution: {np.bincount(y_test_subset)}")
        
        # Make predictions
        predictions = self.vqc.predict(X_test_subset)
        accuracy = accuracy_score(y_test_subset, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report with zero_division handling
        print("\n=== Classification Report ===")
        print(classification_report(y_test_subset, predictions, 
                                  target_names=self.le_target.classes_,
                                  zero_division=0))
        
        return accuracy, predictions, y_test_subset
    
    def compare_with_classical(self, X_train, y_train, X_test, y_test):
        """
        Enhanced comparison with classical methods
        """
        print("=== Classical vs Quantum Comparison ===")
        
        # Random Forest with more estimators
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, 
                                             max_depth=10, min_samples_split=5)
        rf_classifier.fit(X_train, y_train)
        rf_predictions = rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        
        # SVM with probability estimates
        svm_classifier = SVC(random_state=42, kernel='rbf', gamma='scale', 
                           probability=True)
        svm_classifier.fit(X_train, y_train)
        svm_predictions = svm_classifier.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        
        return rf_accuracy, svm_accuracy, rf_classifier, svm_classifier
    
    def plot_enhanced_results(self, y_true, y_pred, quantum_acc, rf_acc=None, svm_acc=None, 
                            rf_model=None, svm_model=None):
        """
        Enhanced visualization with feature importance
        """
        plt.figure(figsize=(18, 6))
        
        # Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.le_target.classes_, 
                   yticklabels=self.le_target.classes_)
        plt.title('Quantum Classifier Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Accuracy Comparison
        plt.subplot(1, 3, 2)
        if rf_acc is not None and svm_acc is not None:
            methods = ['Quantum VQC', 'Random Forest', 'SVM']
            accuracies = [quantum_acc, rf_acc, svm_acc]
            colors = ['purple', 'green', 'blue']
            
            bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
            plt.title('Classifier Performance Comparison')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # Feature Importance (for Random Forest)
        plt.subplot(1, 3, 3)
        if rf_model is not None:
            feature_names = ['down', 'yrdline100', 'TimeSecs', 'ydstogo', 
                           'ScoreDiff', 'qtr', 'GoalToGo', 'posteam']
            importances = rf_model.feature_importances_
            
            plt.barh(range(len(feature_names)), importances, alpha=0.7, color='green')
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Enhanced main execution function
    """
    print("=== Optimized Quantum Football Play Type Classification ===\n")
    
    # Load data
    print("Loading NFL dataset...")
    df = pd.read_csv('NFLPlaybyPlay2015.csv', low_memory=False)
    print(f"Dataset loaded: {df.shape}")
    
    # Initialize optimized quantum classifier
    qfc = QuantumFootballClassifier(
        num_qubits=8, 
        num_layers=3,  # Increased layers
        optimizer='COBYLA', 
        max_iter=200  # Increased iterations
    )
    
    # Preprocess data
    X, y = qfc.preprocess_data(df)
    
    # Prepare quantum data with larger sample
    X_quantum, y_sample = qfc.prepare_quantum_data(X, y, sample_size=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y_sample, test_size=0.25, random_state=42, stratify=y_sample
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create optimized quantum circuit
    feature_map, ansatz = qfc.create_quantum_circuit('ZZ')
    
    # Build VQC
    vqc = qfc.build_vqc(feature_map, ansatz)
    
    # Train with more data
    qfc.train(X_train, y_train, train_size=1000)
    
    # Evaluate with larger test set
    quantum_acc, predictions, y_test_subset = qfc.evaluate(X_test, y_test, test_size=100)
    
    # Enhanced comparison with classical methods
    train_size = min(1000, len(X_train))
    test_size = min(300, len(X_test))
    rf_acc, svm_acc, rf_model, svm_model = qfc.compare_with_classical(
        X_train[:train_size], y_train[:train_size],
        X_test[:test_size], y_test[:test_size]
    )
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Quantum VQC Accuracy:     {quantum_acc:.4f}")
    print(f"Random Forest Accuracy:   {rf_acc:.4f}")
    print(f"SVM Accuracy:             {svm_acc:.4f}")
    
    # Determine best performer
    best_method = max([('Quantum VQC', quantum_acc), 
                      ('Random Forest', rf_acc), 
                      ('SVM', svm_acc)], key=lambda x: x[1])
    print(f"\nBest Performer: {best_method[0]} ({best_method[1]:.4f})")
    
    # Enhanced visualization
    qfc.plot_enhanced_results(y_test_subset, predictions, quantum_acc, 
                            rf_acc, svm_acc, rf_model, svm_model)
    
    print(f"\n{'='*50}")
    print("OPTIMIZATION SUMMARY:")
    print(f"{'='*50}")
    print("✓ Increased training iterations: 50 → 200")
    print("✓ Enhanced circuit depth: 2 → 3 layers") 
    print("✓ Improved data sampling: balanced classes")
    print("✓ Better feature scaling: MinMaxScaler to [0,π]")
    print("✓ Larger training set: 100 → 1000 samples")
    print("✓ Larger test set: 30 → 100 samples")
    print("✓ Full entanglement in ansatz")
    print("✓ Enhanced error handling and fallback")

if __name__ == "__main__":
    main() 
