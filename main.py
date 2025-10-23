import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTableWidget, QTableWidgetItem,
    QWidget, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class AirQualityApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HAVA DURUMU KİRLİLİK TAHMİNİ")
        self.setGeometry(100, 100, 800, 600)

        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Air Quality Categories Label
        self.label_category = QLabel("Air Quality Categories\n Good = 0\n Moderate = 1\n Poor = 2\n Hazardous = 3")
        self.layout.addWidget(self.label_category)

        # ComboBox ve butonlar
        self.dataset_label = QLabel("Veri Seti Seçimi:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Ham Veri", "Gürültülü Veri", "Dengesiz Veri"])
        self.select_dataset_button = QPushButton("Veri Seti Seç")
        self.select_dataset_button.clicked.connect(self.load_dataset)

        self.fillna_label = QLabel("Boşluk Doldurma Yöntemi:")
        self.fillna_combo = QComboBox()
        self.fillna_combo.addItems(["0 ile Doldur", "İleri Doldur", "Geri Doldur", "Ortalama ile Doldur", "Medyan ile Doldur"])
        self.fillna_button = QPushButton("Boşluk Doldur")
        self.fillna_button.clicked.connect(self.fill_missing_values)

        self.model_label = QLabel("Model Seçimi:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["KNN", "Random Forest", "Karar Ağacı", "SVM", "Logistic Regression"])
        self.select_model_button = QPushButton("Model Seç ve Eğit")
        self.select_model_button.clicked.connect(self.train_model)

        self.normalization_label = QLabel("Normalizasyon Yöntemi:")
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["Z-Score", "Min-Max Normalizasyonu"])
        self.normalize_button = QPushButton("Normalizasyon Yap")
        self.normalize_button.clicked.connect(self.normalize_data)

        self.kfold_label = QLabel("K-Fold Sayısı:")
        self.kfold_combo = QComboBox()
        self.kfold_combo.addItems(["2", "3", "4", "5"])
        self.kfold_button = QPushButton("K-Fold Uygula")
        self.kfold_button.clicked.connect(self.apply_kfold)

        # Dengesizlik Giderme Elemanları
        self.balance_label = QLabel("Dengesizlik Giderme Yöntemi:")
        self.balance_combo = QComboBox()
        self.balance_combo.addItems(["Random Over Sampling", "Random Under Sampling", "SMOTE"])
        self.balance_button = QPushButton("Dengesizliği Gider")
        self.balance_button.clicked.connect(self.balance_data)

        self.predict_button = QPushButton("Devam Et")
        self.predict_button.clicked.connect(self.open_prediction_window)

        # Tablo (veri önizleme)
        self.data_table = QTableWidget()

        # Layout'a eklemeler
        self.layout.addWidget(self.dataset_label)
        self.layout.addWidget(self.dataset_combo)
        self.layout.addWidget(self.select_dataset_button)

        self.layout.addWidget(self.fillna_label)
        self.layout.addWidget(self.fillna_combo)
        self.layout.addWidget(self.fillna_button)

        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model_combo)
        self.layout.addWidget(self.select_model_button)

        self.layout.addWidget(self.normalization_label)
        self.layout.addWidget(self.normalization_combo)
        self.layout.addWidget(self.normalize_button)

        self.layout.addWidget(self.kfold_label)
        self.layout.addWidget(self.kfold_combo)
        self.layout.addWidget(self.kfold_button)

        self.layout.addWidget(self.balance_label)
        self.layout.addWidget(self.balance_combo)
        self.layout.addWidget(self.balance_button)

        self.layout.addWidget(self.data_table)
        self.layout.addWidget(self.predict_button)

        # Veri seti ve model için değişkenler
        self.data = None
        self.model = None

    def load_dataset(self):
        dataset_choice = self.dataset_combo.currentText()
        if dataset_choice == "Ham Veri":
            file_path = "C:/Users/Zzz/Desktop/makine_projesi/data.csv"
        elif dataset_choice == "Gürültülü Veri":
            file_path = "C:/Users/Zzz/Desktop/makine_projesi/noisy_data_with_nan.csv"
        elif dataset_choice == "Dengesiz Veri":
            file_path = "C:/Users/Zzz/Desktop/makine_projesi/imbalanced_data.csv"
        else:
            file_path = None

        if file_path:
            data = pd.read_csv(file_path)
            # Air Quality kategorik değerlerini sayısal değere dönüştür
            data['Air Quality'] = data['Air Quality'].map({
                'Good': 0,
                'Moderate': 1,
                'Poor': 2,
                'Hazardous': 3
            })
            self.data = data
            self.display_data()

    def display_data(self):
        if self.data is not None:
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(len(self.data.columns))
            self.data_table.setHorizontalHeaderLabels(self.data.columns)

            for row_idx, row_data in self.data.iterrows():
                self.data_table.insertRow(row_idx)
                for col_idx, cell_data in enumerate(row_data):
                    self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(str(cell_data)))

    def fill_missing_values(self):
        if self.data is not None:
            method = self.fillna_combo.currentText()
            columns_to_fill = [col for col in self.data.columns if col != 'Air Quality']
            if method == "0 ile Doldur":
                self.data[columns_to_fill] = self.data[columns_to_fill].fillna(0)
            elif method == "İleri Doldur":
                self.data[columns_to_fill] = self.data[columns_to_fill].fillna(method='ffill')
            elif method == "Geri Doldur":
                self.data[columns_to_fill] = self.data[columns_to_fill].fillna(method='bfill')
            elif method == "Ortalama ile Doldur":
                for col in columns_to_fill:
                    mean_value = self.data[col].mean()
                    self.data[col] = self.data[col].fillna(mean_value)
            elif method == "Medyan ile Doldur":
                for col in columns_to_fill:
                    median_value = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_value)

            self.display_data()

    def normalize_data(self):
        if self.data is not None:
            method = self.normalization_combo.currentText()
            columns_to_normalize = [col for col in self.data.columns if col != 'Air Quality']

            if method == "Z-Score":
                scaler = StandardScaler()
                self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])
            elif method == "Min-Max Normalizasyonu":
                scaler = MinMaxScaler()
                self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])

            self.display_data()

    def train_model(self):
        if self.data is not None:
            try:
                X = self.data.iloc[:, :-1]
                y = self.data.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model_choice = self.model_combo.currentText()

                if model_choice == "Random Forest":
                    self.model = RandomForestClassifier(random_state=42)
                elif model_choice == "KNN":
                    self.model = KNeighborsClassifier()
                elif model_choice == "Karar Ağacı":
                    self.model = DecisionTreeClassifier(random_state=42)
                elif model_choice == "SVM":
                    self.model = SVC(probability=True, random_state=42)
                elif model_choice == "Logistic Regression":
                    self.model = LogisticRegression(random_state=42)
                else:
                    self.model = None

                if self.model is not None:
                    self.model.fit(X_train, y_train)
                    QMessageBox.information(self, "Model Eğitimi", "Model başarıyla eğitildi.")

                    # Test verisiyle tahmin yap ve metrikleri hesapla
                    y_pred = self.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    cm = confusion_matrix(y_test, y_pred)

                    # Çok sınıflı için her sınıfın özgüllüğü (specificity) hesaplama
                    specificity_list = []
                    for i in range(len(cm)):
                        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                        FP = cm[:, i].sum() - cm[i, i]
                        specificity_list.append(TN / (TN + FP) if (TN + FP) != 0 else 0)
                    specificity = sum(specificity_list) / len(specificity_list)

                    metrics_message = (
                        f"Accuracy: {accuracy:.4f}\n"
                        f"Precision: {precision:.4f}\n"
                        f"Sensitivity (Recall): {recall:.4f}\n"
                        f"Specificity: {specificity:.4f}\n"
                        f"F1 Score: {f1:.4f}"
                    )

                    QMessageBox.information(self, "Model Metrikleri", metrics_message)

                    self.X_test = X_test
                    self.y_test = y_test
                else:
                    QMessageBox.warning(self, "Uyarı", "Model seçilmedi.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Model eğitilirken hata oluştu: {e}")

    def apply_kfold(self):
        """K-Fold'u her fold için metrikleri göstererek uygular."""
        if self.data is not None:
            try:
                X = self.data.iloc[:, :-1]
                y = self.data.iloc[:, -1]
                k = int(self.kfold_combo.currentText())

                model_choice = self.model_combo.currentText()
                # Model seçimine göre yeni bir model örneği oluşturuluyor
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif model_choice == "KNN":
                    model = KNeighborsClassifier()
                elif model_choice == "Karar Ağacı":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_choice == "SVM":
                    model = SVC(probability=True, random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                else:
                    model = None

                if model is not None:
                    from sklearn.model_selection import StratifiedKFold
                    import numpy as np

                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

                    accuracy_list = []
                    precision_list = []
                    recall_list = []
                    f1_list = []
                    specificity_list = []

                    details_info = ""

                    fold_num = 1
                    for train_idx, test_idx in skf.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        # Her fold'da model tekrar eğitilir
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average='macro')
                        rec = recall_score(y_test, y_pred, average='macro')
                        f1 = f1_score(y_test, y_pred, average='macro')

                        cm = confusion_matrix(y_test, y_pred)
                        spec_list = []
                        for i in range(len(cm)):
                            TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                            FP = cm[:, i].sum() - cm[i, i]
                            spec_list.append(TN / (TN + FP) if (TN + FP) != 0 else 0)
                        specificity_fold = np.mean(spec_list)

                        accuracy_list.append(acc)
                        precision_list.append(prec)
                        recall_list.append(rec)
                        f1_list.append(f1)
                        specificity_list.append(specificity_fold)

                        # Her fold'un sonuçlarını satır satır ekleyelim
                        details_info += (
                            f"Fold {fold_num} => "
                            f"Accuracy: {acc:.4f}, "
                            f"Precision: {prec:.4f}, "
                            f"Sensitivity: {rec:.4f}, "
                            f"Specificity: {specificity_fold:.4f}, "
                            f"F1: {f1:.4f}\n"
                        )
                        fold_num += 1

                    # K-Fold bitti, ortalamaları hesapla
                    mean_accuracy = np.mean(accuracy_list)
                    mean_precision = np.mean(precision_list)
                    mean_recall = np.mean(recall_list)
                    mean_f1 = np.mean(f1_list)
                    mean_specificity = np.mean(specificity_list)

                    # Ortalamaları ekleyelim
                    details_info += "------------------------------\n"
                    details_info += (
                        f"Mean Accuracy: {mean_accuracy:.4f}\n"
                        f"Mean Precision: {mean_precision:.4f}\n"
                        f"Mean Sensitivity: {mean_recall:.4f}\n"
                        f"Mean Specificity: {mean_specificity:.4f}\n"
                        f"Mean F1 Score: {mean_f1:.4f}"
                    )

                    QMessageBox.information(self, "K-Fold Sonuçları", details_info)
                else:
                    QMessageBox.warning(self, "Uyarı", "Geçerli bir model seçilmedi.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"K-Fold uygulanırken hata oluştu: {e}")

    def balance_data(self):
        if self.data is not None:
            technique = self.balance_combo.currentText()
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]

            class_counts_before = y.value_counts().to_dict()

            if technique == "Random Over Sampling":
                sampler = RandomOverSampler(random_state=42)
            elif technique == "Random Under Sampling":
                sampler = RandomUnderSampler(random_state=42)
            elif technique == "SMOTE":
                sampler = SMOTE(random_state=42)
            else:
                sampler = None

            if sampler is not None:
                X_res, y_res = sampler.fit_resample(X, y)
                class_counts_after = pd.Series(y_res).value_counts().to_dict()

                balanced_data = pd.DataFrame(X_res, columns=X.columns)
                balanced_data["Air Quality"] = y_res
                self.data = balanced_data
                self.display_data()

                # Mesaj kutusunda gösterilecek mesaj
                message = f"Dengesizlik giderildi ({technique}).\n\nÖnceki Dağılım:\n"
                for cls, count in class_counts_before.items():
                    message += f"Sınıf {cls}: {count}\n"
                message += "\nGiderme Sonrası Dağılım:\n"
                for cls, count in class_counts_after.items():
                    message += f"Sınıf {cls}: {count}\n"

                QMessageBox.information(self, "Dengesizlik Giderme", message)
            else:
                QMessageBox.warning(self, "Uyarı", "Geçerli bir dengesizlik giderme yöntemi seçilmedi.")

    def open_prediction_window(self):
        if self.data is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce veri setini yükleyin.")
            return

        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Karar Ağacı": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42)
        }

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_score = -1
        best_model_name = None
        best_model_instance = None
        for name, mdl in models.items():
            scores = cross_val_score(mdl, X, y, cv=5)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model_instance = mdl

        best_model_instance.fit(X_train, y_train)
        y_pred = best_model_instance.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion Matrix grafiğini göster
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        self.prediction_window = PredictionWindow(self.data, best_model_name, best_model_instance)
        self.prediction_window.show()


class PredictionWindow(QMainWindow):
    def __init__(self, data, best_model_name, best_model):
        super().__init__()
        self.setWindowTitle("Tahmin Penceresi")
        self.setGeometry(150, 150, 800, 600)

        self.data = data
        self.best_model = best_model

        # Merkezi widget ve ana layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Sol kısım
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.inputs = {}
        features = list(data.columns)
        if 'Air Quality' in features:
            features.remove('Air Quality')

        for feature in features:
            lbl = QLabel(feature)
            line_edit = QLineEdit()
            left_layout.addWidget(lbl)
            left_layout.addWidget(line_edit)
            self.inputs[feature] = line_edit

        self.predict_button = QPushButton("Tahmin Yap")
        self.predict_button.clicked.connect(self.make_prediction)
        left_layout.addWidget(self.predict_button)

        main_layout.addWidget(left_widget)

        # Sağ kısım
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.best_model_label = QLabel(f"En iyi model: {best_model_name}")
        self.best_model_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        right_layout.addWidget(self.best_model_label)
        right_layout.addStretch()

        main_layout.addWidget(right_widget)

    def make_prediction(self):
        try:
            input_data = []
            for feature, line_edit in self.inputs.items():
                value = float(line_edit.text())
                input_data.append(value)

            prediction = self.best_model.predict([input_data])
            QMessageBox.information(self, "Tahmin Sonucu", f"Tahmin Edilen Air Quality: {int(prediction[0])}")
        except ValueError:
            QMessageBox.warning(self, "Hata", "Lütfen geçerli sayısal değerler girin.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AirQualityApp()
    window.show()
    sys.exit(app.exec_())
