import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


# Caching raw data loading functions
@st.cache_data
def load_mit_train_data():
    return pd.read_csv("mit_train.csv")


@st.cache_data
def load_mit_test_data():
    return pd.read_csv("mit_test.csv")


@st.cache_data
def load_ptb_normal_data():
    return pd.read_csv("ptb_normal.csv")


@st.cache_data
def load_ptb_abnormal_data():
    return pd.read_csv("ptb_abnormal.csv")


# Caching clean data loading functions
@st.cache_data
def load_mit_train_clean():
    return pd.read_csv("mit_train_clean.csv")


@st.cache_data
def load_mit_test_clean():
    return pd.read_csv("mit_test_clean.csv")


@st.cache_data
def load_ptb_train_clean():
    return pd.read_csv("ptb_train_clean.csv")


@st.cache_data
def load_ptb_test_clean():
    return pd.read_csv("ptb_test_clean.csv")


# Load raw datasets
mit_train_df = load_mit_train_data()
mit_test_df = load_mit_test_data()
ptb_normal_df = load_ptb_normal_data()
ptb_abnormal_df = load_ptb_abnormal_data()

# Load clean datasets
mit_train = load_mit_train_clean()
mit_test = load_mit_test_clean()
ptb_train = load_ptb_train_clean()
ptb_test = load_ptb_test_clean()


@st.cache_data
def preprocess_data(dataset_option, resampling_method, rescaling_method):
    # Split datasets
    # mit
    X_train_mit = mit_train.drop(columns=["target"])
    y_train_mit = mit_train["target"]

    X_test_mit = mit_test.drop(columns=["target"])
    y_test_mit = mit_test["target"]

    # ptb
    X_train_ptb = ptb_train.drop(columns=["target"])
    y_train_ptb = ptb_train["target"]

    X_test_ptb = ptb_test.drop(columns=["target"])
    y_test_ptb = ptb_test["target"]

    # st.write(f"MIT Training set size: {X_train_mit.shape[0]}")
    # st.write(f"MIT Test set size: {X_test_mit.shape[0]}")
    # st.write(f"PTB Training set size: {X_train_ptb.shape[0]}")
    # st.write(f"PTB Test set size: {X_test_ptb.shape[0]}")

    # Select dataset
    if dataset_option == "MIT":
        X_train, y_train = X_train_mit, y_train_mit
        X_test, y_test = X_test_mit, y_test_mit
    else:
        X_train, y_train = X_train_ptb, y_train_ptb
        X_test, y_test = X_test_ptb, y_test_ptb

    # Apply resampling
    resampling_methods = {
        "SMOTE": SMOTE(),
        "Oversampling": RandomOverSampler(sampling_strategy="not majority"),
        "Undersampling": RandomUnderSampler(sampling_strategy="majority"),
        "None": None,
    }

    if resampling_method != "None":
        sampler = resampling_methods[resampling_method]
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Apply rescaling
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "None": None,
    }

    scaler = scalers[rescaling_method]
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


@st.cache_data
def evaluate_model(X_train, y_train, model_option):
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    }
    model = models[model_option]

    skf = StratifiedKFold(n_splits=5)
    f1_scores = []
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        f1_scores.append(f1_score(y_test_fold, y_pred))
    return np.mean(f1_scores)


# Title of the app
st.title("Deep Learning for Heartbeat Analysis: Normal vs. Abnormal")

# Sidebar for navigation
st.sidebar.title("Table of Contents")
pages = [
    "Project Overview",
    "Data Exploration",
    "Data Visualization",
    "Data Preprocessing",
    "Modeling & Evaluation",
    "Model Comparison",
    "Conclusion",
    "References",
]
page = st.sidebar.radio("Go to", pages)

if page == "Project Overview":
    st.header("Project Overview")

    # Layout to place images at the corner
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Introduction")
        st.write(
            """
        This project aims to detect the normality and abnormality of heartbeat 
        rates using advanced machine and deep learning techniques.
        Understanding and accurately identifying abnormal heartbeats can play 
        a crucial role in diagnosing potential heart conditions early.
        """
        )

        st.subheader("Objective")
        st.write(
            """
        The main goal of this project is to develop and evaluate models that 
        can classify heartbeat data as normal or abnormal.
        Users will gain insights into the methods used and the performance of 
        different models.
        """
        )

        st.subheader("Data Source")
        st.write(
            """
        The dataset used for this project is sourced from [this Kaggle dataset]
        (https://www.kaggle.com/datasets/shayanfazeli/heartbeat). 
        It includes comprehensive ECG data and has been preprocessed for
        analysis.
        """
        )

        st.subheader("Techniques Used")
        st.write(
            """
        We employed various machine and deep learning techniques, including 
        logistic regression, decision trees, and neural networks.
        """
        )

        st.subheader("Project Structure")
        st.write(
            """
        The app is divided into several sections:
        - **Data Exploration**: Explore the dataset and key statistics.
        - **Data Preprocessing**: Details on data cleaning and preparation.
        - **Feature Engineering**: Techniques used for feature selection and 
        creation.
        - **Data Visualization**: Visual representation of the data.
        - **Modeling & Evaluation**: Building and assessing the models.
        - **Model Comparison**: Comparing the performance of different models.
        - **Conclusion**: Key findings and future work.
        - **User Guide**: Instructions for using the app.
        - **References**: Sources and references used in the project.
        """
        )

    with col2:
        st.image("medical-cardio.jpg", use_container_width=True)
        st.image("ECG.jpg", use_container_width=True)


if page == "Data Exploration":
    st.header("Data Exploration")

    # Introduction
    st.subheader("Dataset Overview")
    st.write(
        """
        In this experiment, we utilize the freely available Heartbeat dataset 
        from Kaggle,
        curated by Shayan Fazeli. It includes various ECG signals classified 
        into different
        categories such as 'Normal' and various 'Abnormal' heartbeats.
        
        We work with two ECG Heartbeat Categorization Datasets:
        the MIT-BIH Arrhythmia Database (mitbih) and the PTB Diagnostic ECG 
        Database (ptbdb).
        
        The MIT-BIH Arrhythmia Database contains 48 half-hour ECG recordings 
        from 47 individuals recorded
        between 1975-1979. The recordings were digitized at 360 samples per 
        second with 11-bit resolution.
        
        The PTB Diagnostic ECG Database consists of recordings from healthy 
        subjects and patients
        with various heart conditions. It includes 15 signals for each record, 
        with data digitized at 1000 samples per second.
    """
    )

    # Dataset Selection
    dataset_option = st.selectbox(
        "Select a dataset to explore",
        ("MIT Train", "MIT Test", "PTB Normal", "PTB Abnormal"),
    )

    # Function to explore selected dataset
    def explore_dataset(df, dataset_name):
        st.write(f"### {dataset_name} Dataset")
        st.write(f"Number of records: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1]}")
        st.write("Sample data:")
        st.write(df.head())

        # Calculate total missing values
        total_missing = df.isnull().sum().sum()

        # Show total missing values
        show_missing = st.checkbox(f"Show total number of missing values")
        if show_missing:
            st.write(f"Total missing values: {total_missing}")

        show_duplicate = st.checkbox(f"Show number of duplicated rows")
        if show_duplicate:
            st.write(f"Total number of duplicated rows: {df.duplicated().sum()}")

        # Summary information based on dataset selection
        st.write("### Dataset Summary")
        if dataset_name == "MIT Train" or dataset_name == "MIT Test":
            st.write(
                f"Each row represents a single heartbeat, and each column from c-0 to c-{df.shape[1]-2} represents the signal values at different time points. c-{df.shape[1]-1} is the target label for the heartbeat class."
            )
        elif dataset_name == "PTB Normal":
            st.write(
                f"Each row represents a single heartbeat, and each column from c-0 to c-{df.shape[1]-2} represents the signal values at different time points. c-{df.shape[1]-1} is the target label which is 0 for normal heartbeats."
            )
        elif dataset_name == "PTB Abnormal":
            st.write(
                f"Each row represents a single heartbeat, and each column from c-0 to c-{df.shape[1]-2} represents the signal values at different time points. c-{df.shape[1]-1} is the target label which is 1 for abnormal heartbeats."
            )

        # Show detailed information
        show_details = st.checkbox(f"Show more detailed information")
        if show_details:
            if dataset_name == "MIT Train":
                st.write(
                    """
                    **Details of MIT Train Dataset:** 
                    - Size and Structure: The file is quite large, with a size of 392 MB. It contains 87,554 rows, each representing a single heartbeat. 
                    - Columns: There are 188 columns in the file. The first 187 columns represent the signal values at different time points, and the last column is the target label for the heartbeat class. 
                    - Classes: The target column categorizes heartbeats into five classes: 
                      0: Normal heartbeats (“Normal”) 
                      1: Supraventricular premature beats (“Supraventricular”) 
                      2: Ventricular escape beats (“Ventricular”) 
                      3: Fusion of ventricular and normal beats (“Fusion”) 
                      4: Unclassified beats (“Unclassifiable”) 
                    - Class Imbalance: The dataset is known to be imbalanced, with a significant majority of the heartbeats labeled as normal (class “0”). This imbalance needs to be addressed during the model training process to prevent bias. 
                """
                )

            if dataset_name == "MIT Test":
                st.write(
                    """
                    **Details of MIT Test Dataset:** 
                    - Size and Structure: The file is approximately 98.1 MB in size. It contains 21,892 rows, with each row representing a single heartbeat. 
                    - Columns: Similar to the training file, there are 188 columns. The first 187 columns represent the individual time points of the ECG signal, and the final column is the class label. 
                    - Classes: The class labels are the same as in the training set, with five categories ranging from normal heartbeats to various types of arrhythmias. 
                    - Usage: This file is used to evaluate the performance of the model trained on the mitbih_train.csv data. It helps in assessing how well the model generalizes to new, unseen data. 
                """
                )

            if dataset_name == "PTB Normal":
                st.write(
                    """
                    **Details of PTB Normal Dataset:** 
                    - The PTB Diagnostic ECG Database utilizes a specialized recorder with 16 input channels, including 14 for ECGs, one for respiration, and one for line voltage. 
                    - The recorder handles an input voltage of ±16 mV with a high resolution of 16 bits and a bandwidth of 0-1 kHz.
                    - It contains 549 records from 290 subjects, with ages ranging from 17 to 87.
                    - Signal Measurement: Each record has 15 signals, including the standard 12 ECG leads plus 3 Frank lead ECGs, all digitized at 1000 samples per second.
                    - It contains ECG recordings from healthy subjects. The data is structured similarly to the MITBIH files, with each row representing a single heartbeat and columns representing the signal values at different time points. The last column typically contains the label, which in this case would be 0 indicating a normal heartbeat.
                """
                )

            if dataset_name == "PTB Abnormal":
                st.write(
                    """
                    **Details of PTB Abnormal Dataset:** 
                    - The PTB Diagnostic ECG Database utilizes a specialized recorder with 16 input channels, including 14 for ECGs, one for respiration, and one for line voltage. 
                    - The recorder handles an input voltage of ±16 mV with a high resolution of 16 bits and a bandwidth of 0-1 kHz.
                    - It contains 549 records from 290 subjects, with ages ranging from 17 to 87.
                    - Signal Measurement: Each record has 15 signals, including the standard 12 ECG leads plus 3 Frank lead ECGs, all digitized at 1000 samples per second.
                    - It includes ECG recordings from patients with various heart conditions. The structure is the same as the ptbdb_normal.csv file, with the last column containing the label 1 to indicate an abnormal heartbeat.
                """
                )

    if dataset_option == "MIT Train":
        explore_dataset(mit_train_df, "MIT Train")

    elif dataset_option == "MIT Test":
        explore_dataset(mit_test_df, "MIT Test")

    elif dataset_option == "PTB Normal":
        explore_dataset(ptb_normal_df, "PTB Normal")

    elif dataset_option == "PTB Abnormal":
        explore_dataset(ptb_abnormal_df, "PTB Abnormal")


if page == "Data Visualization":
    st.header("Data Visualization")

    st.write(
        """
        In this section, we present various visualizations to help understand the dataset and the results of our analysis.
        The figures below have been pre-generated to save computation time and ensure consistent presentation.
    """
    )

    with st.expander("Raw ECG Signals"):
        st.write(
            """ This visualization aids in understanding the variations and patterns in heartbeat data,
                 crucial for developing accurate machine learning models for heartbeat classification."""
        )

        dataset_option_ecg = st.selectbox(
            "Select a dataset to visualize:",
            ("MIT Dataset", "PTB Dataset"),
            key="ecg",
        )

        if dataset_option_ecg == "MIT Dataset":
            st.markdown("##### Figure 1: Raw ECG Signals for MIT Dataset")
            st.image("MIT_ECG_Signals.png", width=600)
        elif dataset_option_ecg == "PTB Dataset":
            st.markdown("##### Figure 2: Raw normal and abnormal ECG Signals for PTB Dataset")
            st.image("PTB_ECG_Signals.png", width=600)

    with st.expander("The distribution of different heartbeat categories"):
        st.write(
            """ This analysis reveals that the majority of the data consists of normal heartbeats,
                 while the other categories are significantly less frequent. 
                 Understanding this distribution is essential for developing and training machine learning models,
                 as it impacts the model’s ability to accurately classify and detect abnormalities in heartbeats."""
        )

        dataset_option_pie = st.selectbox(
            "Select a dataset to visualize:",
            ("MIT Dataset", "PTB Dataset"),
            key="pie",
        )

        if dataset_option_pie == "MIT Dataset":
            st.markdown(
                "##### Figure 1: The distribution of different heartbeat categories for MIT Dataset"
            )
            st.image("pie_distribution_mit.png", width=600)
        elif dataset_option_pie == "PTB Dataset":
            st.markdown(
                "##### Figure 2: The distribution of different heartbeat categories for PTB Dataset"
            )
            st.image("pie_distribution_PTB.png", width=600)
            
    with st.expander("PCA Dimensionality Reduction"):
        st.write(
            """PCA is a statistical technique used to reduce the dimensionality of a dataset while retaining most of the variation in the data. 
               This can be useful for visualization and for improving the performance of machine learning algorithms."""
        )

        dataset_option_pca = st.selectbox(
            "Select a dataset to visualize PCA:",
            ("MIT Dataset", "PTB Dataset"),
            key="pca",
        )

        if dataset_option_pca == "MIT Dataset":
            st.markdown("##### Scree Plot for MIT Dataset")
            st.image("MIT_screeplot.png", width=400)
            st.markdown("##### PCA Visualization for MIT Dataset")
            st.image("MIT_PCA.png", width=400)
            
        elif dataset_option_pca == "PTB Dataset":
            st.markdown("##### Scree Plot for PTB Dataset")
            st.image("PTB_screeplot.png", width=400)
            st.markdown("##### PCA Visualization for PTB Dataset")
            st.image("PTB_PCA_3d.png", width=400)
    
    with st.expander("t-SNE Dimensionality Reduction"):
        st.write(
            """t-SNE is a machine learning algorithm for visualization that is particularly well suited for embedding high-dimensional data into a space of two or three dimensions.
               This helps in visualizing the structure and clusters in high-dimensional data."""
        )

        dataset_option_tsne = st.selectbox(
            "Select a dataset to visualize t-SNE:",
            ("MIT Dataset", "PTB Dataset"),
            key="tsne",
        )

        if dataset_option_tsne == "MIT Dataset":
            st.markdown("##### t-SNE Visualization for MIT Dataset")
            st.image("MIT_tsne.png", width=600)
        elif dataset_option_tsne == "PTB Dataset":
            st.markdown("##### t-SNE Visualization for PTB Dataset")
            st.image("PTB_tsne.png", width=600)

    


st.markdown(
    """
    <style>
    .highlight {
        background-color: #32CD32;
    }
    </style>
    """, unsafe_allow_html=True
)

if page == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.write(
        """
        In this section, we focus on the MIT dataset for our data preprocessing steps.
        We have chosen to focus on the MIT dataset due to its comprehensive representation and relevance to our study.
        """
    )

    with st.expander("Handling Missing Values"):
        st.write(
            """
            Handling missing values is a critical step in data preprocessing.
            However, in this case, we are fortunate to have no missing values in the MIT dataset.
            Therefore, no imputation or removal of missing values is required at this stage. 
            """
        )

        show_missing_values = st.checkbox("Show missing values")

        if show_missing_values:
            st.write(f"- MIT Train Dataset missing values: {mit_train_df.isnull().sum().sum()}")
            st.write(f"- MIT Test Dataset missing values: {mit_test_df.isnull().sum().sum()}")

    with st.expander("Data Cleaning"):
        st.write(
            """
            Data cleaning involves correcting or removing incorrect, corrupted, or irrelevant parts of the dataset. Steps include:
            """
        )

        show_duplicate = st.checkbox("Show duplicate")
        if show_duplicate:
            st.write(f"- MIT Train Dataset duplicate rows: {mit_train_df.duplicated().sum()}")
            st.write(f"- MIT Test Dataset duplicate rows: {mit_test_df.duplicated().sum()}")

        st.write(
            """
            - Removing duplicate rows to ensure each row represents unique information.
            """
        )

    with st.expander("Split the Data"):
        st.write(
            """
            Before applying any resampling and rescaling techniques, we split the MIT dataset into training and testing sets. 
            This prevents any data leakage and ensures that the model is evaluated on unseen data.
            """
        )
        # Example code for splitting the data
        # st.write(f"MIT Training set size: {X_train_mit.shape[0]}")
        # st.write(f"MIT Test set size: {X_test_mit.shape[0]}")

    with st.expander("Resampling Results"):
        st.write(
            """
            Below are the mean F1-scores for different combinations of resampling methods and models applied to the MIT dataset.
            """
        )

        # Creating the table for resampling results with highlighted cell
        st.markdown(
            """
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>SMOTE</th>
                        <th>OverSampling</th>
                        <th>UnderSampling</th>
                        <th>None</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>LogisticRegression</td>
                        <td>0.408</td>
                        <td>0.408</td>
                        <td>0.407</td>
                        <td>0.408</td>
                    </tr>
                    <tr>
                        <td>Decision Tree</td>
                        <td>0.760</td>
                        <td>0.788</td>
                        <td>0.626</td>
                        <td>0.781</td>
                    </tr>
                    <tr>
                        <td>KNN</td>
                        <td>0.824</td>
                        <td class="highlight">0.835</td>
                        <td>0.752</td>
                        <td>0.835</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True
        )

        st.write(
            """
            **Benefits of Resampling:**
            - Resampling methods such as SMOTE, Oversampling, and Undersampling help address class imbalance in the dataset, which can improve the model's ability to generalize and avoid bias towards the majority class.
            - It is generally advisable to apply resampling methods before rescaling to avoid introducing bias during the scaling process.

            **Note:** After testing various combinations, we have found that Oversampling works best for our dataset.
            """
        )

    with st.expander("Rescaling Results"):
        st.write(
            """
            Below are the mean F1-scores for different combinations of rescaling methods and models applied to the MIT dataset.
            """
        )

        # Creating the table for rescaling results with highlighted cell
        st.markdown(
            """
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>StandardScaler</th>
                        <th>MinMaxScaler</th>
                        <th>RobustScaler</th>
                        <th>None</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>LogisticRegression</td>
                        <td>0.408</td>
                        <td>0.408</td>
                        <td>0.409</td>
                        <td>0.408</td>
                    </tr>
                    <tr>
                        <td>Decision Tree</td>
                        <td>0.781</td>
                        <td>0.781</td>
                        <td>0.781</td>
                        <td>0.781</td>
                    </tr>
                    <tr>
                        <td>KNN</td>
                        <td>0.847</td>
                        <td class="highlight">0.855</td>
                        <td>0.846</td>
                        <td>0.854</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True
        )

        st.write(
            """
            **Benefits of Rescaling:**
            - Rescaling methods such as StandardScaler, MinMaxScaler, and RobustScaler ensure that features have comparable scales, which helps improve the performance and convergence rate of the models.
            - It is generally advisable to apply rescaling methods after resampling to avoid introducing bias during the scaling process.

            **Note:** After testing various combinations, we have found that MinMaxScaler works best for our dataset.
            """
        )

if page == "References":
    st.header("References")
    st.write(
        """
        Here are the references used in this project:
        1. [GitHub: Heartbeat Analysis AI](https://github.com/Ping-YUAN/heartbeat-analysis-ai)
        2. [Kaggle: Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
        3. Mark RG, Schluter PS, Moody GB, Devlin, PH, Chernoff, D. (1982). An annotated ECG database for evaluating arrhythmia detectors. IEEE Transactions on Biomedical Engineering, 29(8), 600.
        4. Moody GB, Mark RG. (1990). The MIT-BIH Arrhythmia Database on CD-ROM and software for use with it. Computers in Cardiology, 17, 185-188.
        5. Moody GB, Mark RG. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology, 20(3), 45-50. (PMID: 11446209)
        6. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101(23), e215–e220.
        
    """
    )


if page == "Conclusion":
    st.header("Conclusion")

    st.subheader("Summary of Findings")
    st.write(
        """
    Our modeling efforts demonstrated strong classification performance across both the MIT and PTB datasets, with the DNN model achieving the highest accuracy. Through systematic hyperparameter tuning, we optimized the KNN, Decision Tree, and XGBoost models, with XGBoost achieving solid performance on the PTB dataset, but less so on MIT.
    """
    )

    st.subheader("Interpretability Analysis")
    st.write(
        """
    Our interpretability analysis, using SHAP and LIME, highlighted critical features influencing predictions, particularly early-cycle features in the heartbeat sequence that proved significant in the DNN’s decision-making. These insights suggest that key segments of the heartbeat carry essential signals for classification, which the models leverage effectively. The local interpretability provided by LIME offered additional context for individual predictions, aligning well with global feature importance in many cases.
    """
    )

    st.subheader("Model Performance and Future Work")
    st.write(
        """
    While the DNN's architecture proved effective for both datasets, further tuning is needed for LSTM and Transformer models due to their unexpectedly low performance. Given differences in feature importance across datasets, merging the MIT and PTB datasets for a combined analysis may enhance model generalizability and robustness. This integration will support a more comprehensive interpretability framework, enabling a deeper understanding of feature contributions across diverse heartbeat signals.
    """
    )

    st.subheader("Project Overview")
    st.write(
        """
    In this report, we use advanced machine learning and deep learning techniques to detect whether a heartbeat is normal or abnormal. It aims to improve the accuracy of heartbeat detection and classification within general supervision by using both traditional machine learning techniques, as well as more advanced deep learning models. We provide a comprehensive overview of the methodologies employed, including data exploration, data preprocessing, feature engineering, model selection, and evaluation metrics.
    """
    )

    st.subheader("Conclusion and Future Directions")
    st.write(
        """
    Overall, our results show promising efficacy of these techniques to improve diagnostic accuracy and possibly help in early diagnosis of cardiac anomalies. Supported by expert mentorship, our group work has resulted in detailed discussions and encouraging outcomes that could serve as groundwork for further research on medical diagnosis.
    """
    )
