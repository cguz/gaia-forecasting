import os
import time
from typing import Optional, List

# Libraries

## Dummy Output class
class DummyOutput:
    """
    A dummy class to simulate kfp.dsl.Output[Artifact] for local testing.
    """

    def __init__(self, artifact_name: str):
        self.artifact_name = artifact_name
        self.path = f"src/aiqnv_general_ml_pipeline/output/{artifact_name}" 
        self.metadata = {}  

        # create the output directory
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __repr__(self):
        return f"DummyOutput(name={self.artifact_name}, path={self.path}, metadata={self.metadata})"

############################################
########## Components KFP ##################
############################################

#####################################
############# STEP 1.1 ##############
#####################################

def request_location_data_catalog(
    request_to: str, type_data: Optional[str] = None, mission: Optional[str] = None, metadata_output: "Output[Artifact]" = None
):
    """
    Request metadata from the data catalog and filter it by mission.

    Args:
        type_data (str): The type of data to request.
        mission (Optional[str]): The mission parameter, optional.

    Returns:
        metadata_output (Output[str]): A json file with information of that contains mission and (provider_name or requirement link, which is url:port).
    """
    print(f"Step 1.1: request data's location of {request_to} {type_data}")

    import json
    from aiqnv_general_ml_pipeline.data_process_and_quality import DataHubAccess

    try:
        # Instantiate the correct class (Missions and Requirements)
        data_catalog_class = DataHubAccess[request_to].value
        data_catalog = data_catalog_class()

        # request the metadata
        metadata = data_catalog.request(type_data, mission)

        if metadata is None:
            print("Error: Metadata is None. Check request method.")
            return None

        if mission:
            metadata_output.metadata["mission"] = mission
        
        if type_data:
            metadata_output.metadata["type"] = type_data

        metadata_str = json.dumps(metadata)
    
        with open(metadata_output.path, "w") as f:
            f.write(metadata_str)

        # TODO: remove - only for local
        return metadata_output
        
    except KeyError:
        print(f"Error: Invalid request_to value: {request_to}. Must be 'Missions' or 'Requirements'.")


def filter_provider_by_mission(metadata_path: "Input[Artifact]", mission: str) -> str:
    """
    Filters the provider name from the metadata based on the mission.

    Args:
        metadata_path: Path to the json file containing metadata information.
        mission: The mission name to filter by.

    Returns:
        The provider name associated with the mission.
        Raises ValueError if no provider is found for the given mission.
    """
    import json

    with open(metadata_path.path, "r") as f:
        metadata_list = json.load(f)

    for entry in metadata_list:
        if entry.get("mission") == mission:
            return entry.get("provider_name")
    raise ValueError(f"No provider found for mission: {mission}")

#####################################
############# STEP 2.1 ##############
#####################################

def download_dataset(
    tm_parameters: List[str],
    from_date: str,
    to_date: str,
    mission: str,
    provider_name: str,
    input_dataset: str,
    output_data: "Output[Dataset]" = None,
):
    """Download telemetry data and store it in TileDB format."""
    import os
    import pandas as pd
    from datetime import datetime

    from aiqnv_general_ml_pipeline.data_process_and_quality import RequestManager, DataManagement
    
    print(f"2.1 Download Mission Data and save with TileDB. For parameters: {tm_parameters}")
    
    # Initialize RequestManager
    request_manager = RequestManager(provider_name=provider_name, source_key=mission)

    # Load data
    data_frames = []

    for param in tm_parameters:
        param_data = request_manager.load_parameter_data(
            parameters=[param], from_date=from_date, to_date=to_date
        )
        
        name = param_data[0]['name']
        value_list = param_data[0]['value']

        rows = []
        for item in value_list:
            timestamp = datetime.fromtimestamp(int(item['date']) / 1000)
            rows.append({name: item['value'], 'Timestamp': timestamp})

        # create dataframe
        param_df = pd.DataFrame(rows)
        data_frames.append(param_df)

    # Combine data into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.to_parquet(output_data.path)
    
    print(f"Data combined: {combined_df}")    
    print(combined_df['Timestamp'])

    # Store data with TileDB
    data_management = DataManagement(dataset_path=input_dataset, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))
    data_management.update(combined_df)


#####################################
############# STEP 3.1 ##############
#####################################

def data_lineage_datacatalog(previous: str, current: "Input[Artifact]", description: str):
    print(f"Step 3.1: {description} - lineage previous: {previous}, current: {current}")

#####################################
############# STEP 3.2 ##############
#####################################
def data_preprocessing(
    input_dataset: str,
    input_generate_failure: bool,
    output_dataframe: "Output[Dataset]",
):
    
    print("Step 3.2: Data processing")
    
    from aiqnv_general_ml_pipeline.data_process_and_quality import DataManagement, TelemetryProcessor
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    COL_TIME = "Timestamp"
    COL_FEATURE = "NEI00026"

    # Load data with TileDB
    data_management = DataManagement(dataset_path=input_dataset, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))
    df = data_management.load()

    print(df.head())
    
    # Perform data pre-processing
    telemetry_processor = TelemetryProcessor()

    if input_generate_failure:

        # Create a proper timestamp object
        date_string = "2021-05-01 01:02:59.866"

        # Parse the date string into a datetime object
        try:
            timestamp = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f") 
        except ValueError:
            try:
                timestamp = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") 
            except ValueError as e:
                print(f"Error parsing date string: {e}")
                raise 

        # Failure Id-1 : out of ranges
        dm_02 = {COL_FEATURE: 50.0000, COL_TIME: timestamp}
        df_preprocessed = pd.concat([df, pd.DataFrame([dm_02])], ignore_index=True)

        # Failure Id-2 : null value
        dm_03 = {COL_FEATURE: np.nan, COL_TIME: timestamp}
        df_preprocessed = pd.concat([df_preprocessed, pd.DataFrame([dm_03])], ignore_index=True)

        # Failure Id-3 : Duplicate value
        df_preprocessed = pd.concat([df_preprocessed, df.iloc[[2]]], ignore_index=True)

        print(df_preprocessed.tail(3))

    else:
        
        # Perform data cleaning, handling NaNs, removing duplicates
        df_preprocessed = telemetry_processor.remove_nan_samples(df).drop_duplicates(subset=[COL_TIME, COL_FEATURE])
        
        # Remove out of ranges values. Note : the lower_bound and upper_bound should be considered from the requirements.
        df_preprocessed = telemetry_processor.remove_out_of_ranges(df_preprocessed, lower_bound=29, upper_bound=34, col=COL_FEATURE)
    
    # Store data in TileDB
    data_management.update(df_preprocessed)

    # Store in the output path
    df_preprocessed.to_parquet(output_dataframe.path)


    #####################################
    
    df_preprocessed.to_csv(f"{output_dataframe.path}.csv")

    # Use fragment_info directly on the URI, which works for any array type
    # fragment_infos = data_management.array_fragments()

    # Access fragment details
    # for fr in fragment_infos:
    #    print(fr)


    # Analyze the data sampling frequency
    # time_diffs = data['ds'].diff().dropna()

    # print("Time differences:")
    # print(time_diffs)

    # print("\nTime differences in seconds:")
    # time_diffs_sec = time_diffs.dt.total_seconds()
    # print(time_diffs_sec)

    # print("\nDescription of the time differences in seconds:")
    # print(time_diffs_sec.describe())


    # mode = time_diffs_sec.mode()
    # print(f'\nMost frequent value for time difference in seconds: {mode=}')
    # mean = time_diffs_sec.mean()
    # print(f'\nMean value for time difference in seconds: {mean=}')


#####################################
############# STEP 2.2 ##############
#####################################
def request_requirements(url: str, w_step: str, output_requirements: "Output[Artifact]"):
    print("Step 2.2: request requirements")
    
    import requests
    import json

    #     # Send a GET request to the API with the "w_step" parameter in the query string.
    #     requirements = requests.get(url, params={"w_step": w_step}).json()

    #     # Access requirements details
    #     for req in requirements:
    #         print(f"ID: {req.get('id')}")
    #         print(f"ML requirement: {req.get('ml_requirement')}")
    #         print(f"W step: {req.get('w_step')}")
    #         print(f"Type: {req.get('type')}")
    #         print(f"Parameters: {req.get('params')}")
    #         print("-" * 50)

    # Mocked requirements
    requirements = [
        {
            "id": "REQ-001",
            "ml_requirement": "NEI00026 should be higher than 0",
            "w_step": "Data Validation",
            "type": "higher_than",
            "params": {"column_name": "NEI00026", "number": 0},
        },
        {
            "id": "REQ-002",
            "ml_requirement": "NEI00026 should be below 100",
            "w_step": "Data Validation",
            "type": "smaller_than",
            "params": {"column_name": "NEI00026", "number": 100},
        },
        {
            "id": "REQ-003",
            "ml_requirement": "NEI00026 should be between 20 and 40",
            "w_step": "Data Validation",
            "type": "is_in_range",
            "params": {"column_name": "NEI00026", "min_value": 20, "max_value": 40},
        },
        {
            "id": "REQ-004",
            "ml_requirement": "Model MSE should be higher 0,3",
            "w_step": "Train Model",
            "type": "higher_than",
            "params": {"column_name": "mse", "number": 0.3},
        },
        {
            "id": "REQ-005",
            "ml_requirement": "Model MSE should be higher 0,3",
            "w_step": "Evaluate Model",
            "type": "higher_than",
            "params": {"column_name": "mse", "number": 0.3},
        },
    ]

    filtered_requirements = [
        req for req in requirements if req.get("w_step") == w_step
    ]

    # Write the requirements to the output file
    with open(output_requirements.path, "w") as f:
        f.write(json.dumps(filtered_requirements))

    return output_requirements


#####################################
############# STEP 4.1 ##############
#####################################
def data_check_great_expectations(
    input_dataset: str,
    input_requirements: "Input[Artifact]",
    output_report: "Output[Artifact]",
) -> str:
    print("Step 4.1: Data check with GE")

    import os
    import json

    from aiqnv_general_ml_pipeline.data_process_and_quality import DataQualificationGreatExpectations, DataManagement
    
    # Load data as a dataframe with TileDB
    data_management = DataManagement(dataset_path=input_dataset, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))
    df = data_management.load()

    # Read the requirements from the input artifact
    with open(input_requirements.path, "r") as f:
        requirements = json.load(f)

    # Initialize DataQualificationGreatExpectations with dataframe and requirements
    # It sets the context and creates the checkpoint
    dq = DataQualificationGreatExpectations(df, requirements)

    validation_results = dq.run_validation()

    # output_path = "/tmp/kfp/outputs/Output"
    output_path = "./src/aiqnv_general_ml_pipeline/output/great_expectations_validation_results_summary"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the validation summary to the output file
    with open(output_path, "w") as f:
        f.write(str(validation_results))

    # Write the validation results to the output file
    with open(output_report.path, "w") as f:
        f.write(json.dumps(validation_results))

#####################################
############# STEP 4.2 ##############
#####################################

def data_check_whylogs(
    input_dataset: str,
    input_requirements: "Input[Artifact]",
    profile_output: "Output[Artifact]",
    constraints_report: "Output[Artifact]",
    profile_summary: "Output[Artifact]",
    output_report: "Output[Artifact]",
):

    print("Step 4.2: Data check with WL")

    import os
    import json
    from whylogs.viz import NotebookProfileVisualizer

    from aiqnv_general_ml_pipeline.data_process_and_quality import DataQualificationWhylogs, DataManagement
    
    # Load data as a dataframe with TileDB
    data_management = DataManagement(dataset_path=input_dataset, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))
    df = data_management.load()

    # Read the requirements from the input artifact
    with open(input_requirements.path, "r") as f:
        requirements = json.load(f)

    # Initialize DataQualificationWhylogs with dataframe and requirements.
    # It sets the constraints and profile_view attributes
    dq = DataQualificationWhylogs(df, requirements)

    # Save the profile generated by WhyLogs to a file
    with open(profile_output.path, "wb") as f:
        f.write(dq.profile_view.serialize())

    validation_results = dq.run_validation()

    # Write the validation results to the output file
    with open(output_report.path, "w") as f:
        f.write(json.dumps(validation_results))

    # Generate the constraints report
    visualization = NotebookProfileVisualizer()
    html_constraints_report = visualization.constraints_report(
        dq.constraints, cell_height=300
    )

    # Save the constraints report HTML to a file
    with open(constraints_report.path, "w") as f:
        f.write(html_constraints_report.data)

    # Set the dataset as the target profile
    visualization.set_profiles(target_profile_view=dq.profile_view)

    # Generate a profile summary report
    html_profile_summary = visualization.profile_summary()

    # Save the profile summary report HTML to a file
    with open(profile_summary.path, "w") as f:
        f.write(html_profile_summary.data)


#####################################
############# STEP 5.3 ##############
#####################################

def quality_label_datacatalog(data: str, parameters: str):
    print("Step 5.3: Mark Quality label in datahub")


#####################################
############# STEP 5.6 ##############
#####################################

def split_data(input_dataset: str,
    training_output: "Output[Artifact]",
    test_output: "Output[Artifact]"):
    print("Step 5.6: Split data")

    from aiqnv_general_ml_pipeline.data_process_and_quality import DataManagement

    # Load data with TileDB
    data_management = DataManagement(dataset_path=input_dataset, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))
    df = data_management.load()

    # order data by column 1
    df_sorted = df.sort_values(by=df.columns[1])

    # calculate the train size and split data
    train_size = int(len(df_sorted) * 0.8)
    train = df_sorted.iloc[:train_size]
    test = df_sorted.iloc[train_size:]

    # Write the training data to the output file
    train.to_parquet(training_output.path) 
    
    # Write the test data to the output file
    test.to_parquet(test_output.path) 


#####################################
############# STEP 6.2 ##############
#####################################
def training_ml_model(
    input_data: "Input[Dataset]", 
    input_requirements: "Input[Dataset]", 
    input_mlflow_run_name: str,
    output_model: "Output[Model]", 
    is_valid_output: "OutputPath(bool)") -> None:
    
    print("Step 6.2: Train ML model")
    
    from aiqnv_general_ml_pipeline.model_training_and_quality import TimeSeriesML, CheckModel

    # NOTE: RUN MLFLOW SERVER
    # mlflow server --host 127.0.0.1 --port 8080
    # Configure the MLflow tracking URI to point to the MLflow server
    # remote_server_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
    remote_server_uri = "http://127.0.0.1:8080"

    # Load the data
    data = pd.read_parquet(input_data.path)

    # Hyperparameters
    hyperparameters = {
        "input_size": 10,
        "h": 2000,
        "n_freq_downsample": [2, 1, 1, 1],
        "mlp_units": [[256, 256], [256, 256]],
        "scaler_type": 'standard',
        "max_steps": 10,
        "random_seed":42,  
        #"enable_progress_bar":False,
    }

    # Instantiate the class
    trainer = TimeSeriesML(
        mlflow_experiment_name="generic-pipeline-v2",
        mlflow_run_name=input_mlflow_run_name,
        mlflow_tracking_uri=remote_server_uri,
    )

    # Train and validate the model, storing in mlflow
    _, model_metadata = trainer.train(model_name="nhits", data=data, hyperparameters=hyperparameters, output_path=output_model.path)

    check_model = CheckModel()
    result_check = check_model.check_metrics_model(model_metadata, input_requirements.path)
    
    # Write the boolean value to the specified path
    with open(is_valid_output.path, "w") as f:
        f.write(str(result_check))


#####################################
############# STEP 8 ##############
#####################################

def evaluate_ml_model(
    input_data: "Input[Dataset]",
    input_model: "Input[Model]",
    input_requirements: "Input[Dataset]", 
    input_mlflow_run_name: str,
    output_report: "Output[Artifact]",
    is_valid_output: "OutputPath(bool)"
):

    print("Step 8: Evaluate ML model")

    from aiqnv_general_ml_pipeline.model_training_and_quality import TimeSeriesML, CheckModel

    # NOTE: RUN MLFLOW SERVER
    # mlflow server --host 127.0.0.1 --port 8080
    # Configure the MLflow tracking URI to point to the MLflow server
    # remote_server_uri = "http://mlflow.mlflow.svc.cluster.local:5000"
    remote_server_uri = "http://127.0.0.1:8080"

    # Instantiate the class
    trainer = TimeSeriesML(
        mlflow_experiment_name="generic-pipeline-v2",
        mlflow_run_name=input_mlflow_run_name,
        mlflow_tracking_uri=remote_server_uri,
    )
    
    # Load the data
    data = pd.read_parquet(input_data.path)

    # Evaluate the model, storing in mlflow
    metadata = trainer.evaluate(input_model=input_model.path, input_data=data, output_path=output_report.path)

    check_model = CheckModel()
    result_check = check_model.check_metrics_model(metadata, input_requirements.path)
    
    # Write the boolean value to the specified path
    with open(is_valid_output.path, "w") as f:
        f.write(str(result_check))


#####################################
############# STEP 10 ##############
#####################################
def inference_validation(
    reference_data: "Input[Dataset]",
    analysis_data: "Input[Dataset]",
    drift_report: "Output[Artifact]"
):
    """
    Validate model inference using NannyML for drift detection.

    Args:
        reference_data: Training/reference data
        analysis_data: Production/analysis data
        drift_report: Output report containing drift analysis
    """
    print("Step 10: Validating inference with NannyML")

    import json
    import nannyml as nml
    from aiqnv_general_ml_pipeline.model_production_and_quality import InferenceValidator

    # Create UnivariateDriftCalculator
    column_names = ['car_value', 'salary_range', 'debt_to_income_ratio', 'loan_length', 'repaid_loan_on_prev_car', 'size_of_downpayment', 'driver_tenure', 'y_pred_proba', 'y_pred']
    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        treat_as_categorical=['y_pred'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    )

    validator = InferenceValidator(calc)

    results = validator.calculate_drift(reference_data, analysis_data)

    report = validator.generate_report(results)

    # Save drift report
    with open(drift_report.path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Drift analysis report saved to {drift_report.path}")

############################################
############# ML Pipeline ##################
############################################

#########################
## Configurate variables
#########################

tm_parameters = ["NEI00026"]
dataset_path= "test_datasetv5"
expectations_path= ""
timetravel_timestamp = int(time.time() * 1000)


mission = "GAIA"
from_date = "2021-05-01 01:00:00"
to_date = "2021-05-02 01:00:00"
data_management_step = "Data Management"
training_step = "Train Model"
evaluation_step = "Evaluate Model"
production_step = "Production Model"


########################################
## Step 1.1: Get location of mission data
########################################

metadata_output = DummyOutput("location_mission_data_metadata.json")
metadata = request_location_data_catalog("Missions", "tm", "GAIA", metadata_output)

# Filter Provider by Mission using the helper function
provider_name = filter_provider_by_mission(
    metadata_path=metadata, mission=mission
)

print(f"Provider: {metadata}\n")

############################
## Step 2.1: Download Dataset
############################

# Commented because we can not execute it in local
""" download_task = (
    download_dataset(
        tm_parameters=tm_parameters,
        from_date=from_date,
        to_date=to_date,
        mission=mission,
        provider_name=provider_name,
        input_dataset=input_dataset,
        output_data=output_download_dataframe
    )
) """

download_task_output = DummyOutput("download_data")

import pandas as pd
import datetime

combined_df = pd.read_parquet(download_task_output.path)

from aiqnv_general_ml_pipeline.data_process_and_quality import DataManagement

# Store data with TileDB
data_manager = DataManagement(dataset_path=dataset_path, use_minio=False, minio_access_key=os.getenv("MINIO_ACCESS_KEY"), minio_secret_key=os.getenv("MINIO_SECRET_KEY"))

# Process and store data
data_manager.update(combined_df)

print(f"\n{download_task_output}")

combined_df.to_csv(f"{download_task_output.path}.csv")

############################
## Step 3.1: Data Lineage - TBD
############################

data_lineage_datacatalog(mission, download_task_output, "\nStep 3.1: add mission data")

###############################
## Step 3.2: Data Preprocessing
###############################

data_preprocessing_task_output = DummyOutput("preprocessed_data")
data_preprocessing_task = data_preprocessing(
    input_dataset=dataset_path,
    input_generate_failure=True,
    output_dataframe=data_preprocessing_task_output
)

#########################
## Step 4.3: Data Lineage - TBD
#########################

data_lineage_datacatalog("download task", data_preprocessing_task_output, "\nStep 4.3: add pre-processed data")



#########################################
## Step 1.2: Get location of requirements
#########################################

metadata_output = DummyOutput("location_requirements_metadata.json")
metadata = request_location_data_catalog("Requirements", None, "GAIA", metadata_output)

if metadata:
    with open(metadata.path) as f:
        file_content = f.read()
        print(f"Location of management requirement: {file_content}")


#########################################################
## Step 2.2: Download Requirements w-step data management
#########################################################

# Filter the url of the requirement using the helper function
requirements_url = filter_provider_by_mission(
    metadata_path=metadata, mission=mission
)
print(f"url requirements: {requirements_url}\n")

requirements_output = DummyOutput("requirements.json")
request_requirements_task = request_requirements(
    url=requirements_url,
    w_step=data_management_step,
    output_requirements=requirements_output
)


#############
# Step 4.1: Data Check with Great Expectations
#############

data_check_great_expectations_task_output = DummyOutput("great_expectations_task_output.json")

check_ge_task = data_check_great_expectations(
    input_dataset=dataset_path,
    input_requirements=requirements_output,
    output_report=data_check_great_expectations_task_output,
)

# ############
# # Step 4.2: Data Check with WhyLogs
# ############

data_check_whylogs_task_profile_output = DummyOutput("data_check_whylogs_task_profile_output")
data_check_whylogs_task_constraints_report = DummyOutput("data_check_whylogs_task_constraints_report.html")
data_check_whylogs_task_profile_summary = DummyOutput("data_check_whylogs_task_profile_summary.html")
data_check_whylogs_task_output_report = DummyOutput("data_check_whylogs_output_report.json")

check_wl_task = data_check_whylogs(
        input_dataset=dataset_path,
        input_requirements=requirements_output,
        profile_output=data_check_whylogs_task_profile_output,
        constraints_report=data_check_whylogs_task_constraints_report,
        profile_summary=data_check_whylogs_task_profile_summary,
        output_report=data_check_whylogs_task_output_report
)


# # #########################
# # # Step 5.1: Data Lineage
# # #########################
# data_lineage_datacatalog(previous="data_preprocessing_task", current=check_ge_task, description="add GE results")

# # #########################
# # # Step 5.2: Data Lineage
# # #########################
# data_lineage_datacatalog(previous="data_preprocessing_task", current=check_wl_task, description="add WL results")

# # with dsl.If(check_wl_task.outputs['is_valid_output'] == False, name="Whylogs validation"):
    
#     # Step 5.3: Quality label - mark pre-processed data as bad

#     # Stop

# # Else:

# # ###########################################################
# # # Step 5.4: Quality label - mark pre-processed data as good
# # ###########################################################
# quality_label_datacatalog(data = "", parameters = "")

# ######################
# # Step 5.6: Split Data 
# ######################
training_output = DummyOutput("training-data.parket")
test_output = DummyOutput("test-data.parket")
split_data_task = split_data(input_dataset=dataset_path, training_output=training_output, test_output=test_output)

# ########################
# # Step 6.1: Data Lineage
# ########################
data_lineage_datacatalog(previous="GE and WL", current=split_data_task, description="add training and testing data")

# #################################################
# # Step 5.5: Request Requirements w-step training
# #################################################
requirements_training_output = DummyOutput("requirements-training.json")
request_requirements_task = request_requirements(
    url=requirements_url,
    w_step=training_step,
    output_requirements=requirements_training_output
)

#####################################################################

# # ##########################
# # # Step 6.2: Train ML model
# # ##########################
check_is_valid_training_model_output = DummyOutput("check_is_valid_train_model_output")
model_training_output = DummyOutput("model-training-output")
input_mlflow_run_name=mission + "_NHITS_training"
train_ml_model_task = training_ml_model(
     input_data=training_output, 
     input_requirements=requirements_training_output, 
     input_mlflow_run_name=input_mlflow_run_name,
     output_model=model_training_output, 
     is_valid_output=check_is_valid_training_model_output
)


# #########################
# # Step 7.1: Data Lineage
# #########################
# data_lineage_datacatalog(previous="split_data_task", current=train_ml_model_task, description="add trained model and metrics")

# # TODO: TBC if we need a new component to check the training model against the requirements

# # with dsl.If(train_ml_model_task.outputs['is_valid_output'] == False, name="Train model check"):

#     # Step 7.2: Quality label - mark trained model as bad

#     # Stop

# # Else:

# #######################################################
# # Step 7.3: Quality label - mark trained model as good
# #######################################################
quality_label_datacatalog(data = "", parameters = "")

# ###################################################
# # Step 7.4: Request Requirements w step evaluation 
# ###################################################
requirements_evaluation_output = DummyOutput("requirements-evaluation.json")
request_requirements_task = request_requirements(
    url=requirements_url,
    w_step=evaluation_step,
    output_requirements=requirements_evaluation_output
)

# #####################
# # Step 8: Evaluate ML
# #####################
check_is_valid_evaluate_model_output = DummyOutput("check_is_valid_evaluate_model_output")
evaluate_output = DummyOutput("model-evaluate-output")
input_mlflow_run_name=mission + "_NHITS_evaluate"
evaluate_ml_model_task = evaluate_ml_model(
     input_data=test_output, 
     input_model=model_training_output, 
     input_requirements=requirements_evaluation_output, 
     input_mlflow_run_name=input_mlflow_run_name,
     output_report=evaluate_output, 
     is_valid_output=check_is_valid_evaluate_model_output
)

# #########################
# # Step 9.1: Data Lineage
# #########################
data_lineage_datacatalog(previous="train_tasks", current=evaluate_ml_model_task, description="add evaluated metrics")


# # TODO: TBC if we need a new component to check the evaluate model against the requirements, maybe same as the training model

# # with dsl.If(evaluate_ml_model_task.outputs['is_valid_output'] == False, name="Evaluate model check"):

#     # Step 9.2: Quality label - mark evaluated model as bad

#     # Stop

# # Else:

# #########################################################
# # Step 9.3: Quality label - mark evaluated model as good
# #########################################################
quality_label_datacatalog(data = "good", parameters = "")

# #############
# # Step 9.5: Download and Serve latest (production) ML model with KServe
# #############
# serving_ml_model_task = serve_ml_model_kserve(
#     model=train_ml_model_task.outputs["output_model"]
# ).after(evaluate_ml_model_task)

# ###################################################
# # Step 9.4: Request Requirements w-step production
# ###################################################
requirements_production_output = DummyOutput("requirements-production.json")
request_requirements_task = request_requirements(
    url=requirements_url,
    w_step=production_step,
    output_requirements=requirements_production_output
)

#############################################
# Step 10: Inference Validation with NannyML
#############################################

##############################################################################
# Remove this section when we have a real reference_data and analysis_data
# and update the inference_validation call
import nannyml as nml
mock_test_output, mock_evaluate_ml_output, _ = nml.load_synthetic_car_loan_dataset()
##############################################################################

drift_report_output = DummyOutput("drift_report_output")

inference_validation_task = inference_validation(
    reference_data=mock_test_output, #test_output
    analysis_data=mock_evaluate_ml_output, #evaluate_ml_output
    drift_report=drift_report_output
)

# #############
# # Step 11.1: Data Lineage
# #############
# data_lineage_datacatalog(previous="evaluation model", current=inference_validation_task, description="add nannyml results")


# # with dsl.If(inference_validation_task.outputs['is_valid_output'] == False, name="Inference check"):

# #############
# # Step 11.2: Quality label - mark evaluated model as bad
# #############
# quality_label_datacatalog(data = "good or bad", parameters = "")
