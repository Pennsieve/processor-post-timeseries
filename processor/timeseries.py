import os
import json
import boto3
import requests

def authenticate():

    PENNSIEVE_URL = "https://api.pennsieve.io"
    email = os.getenv("PENNSIEVE_API_KEY")
    password = os.getenv("PENNSIEVE_API_SECRET")

    r = requests.get(f"{PENNSIEVE_URL}/authentication/cognito-config")
    r.raise_for_status()

    cognito_app_client_id = r.json()["tokenPool"]["appClientId"]
    cognito_region = r.json()["tokenPool"]["region"]

    cognito_client = boto3.client(
        "cognito-idp",
        region_name=cognito_region,
        aws_access_key_id=email,
        aws_secret_access_key=password,
    )

    login_response = cognito_client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": email, "PASSWORD": password},
        ClientId=cognito_app_client_id
    )

    api_key = login_response["AuthenticationResult"]["AccessToken"]
    return api_key

def getWorkflowData(session_key):
    integration_id = os.getenv("INTEGRATION_ID")
    PENNSIEVE_URL_2 = "https://api2.pennsieve.io"

    r = requests.get(f"{PENNSIEVE_URL_2}/workflows/instances/{integration_id}", headers={"Authorization": f"Bearer {session_key}"})

    return json.loads(r.text)

import requests

def getBDFPackageId(session_key, dataset_id):
    ''' 
    Gets the package ID of the file with the .bdf extension in the specified dataset. 
    Only expects one package with this extension. 
    If multiple packages are found, an error is raised.
    
    Parameters:
        session_key (str): Bearer token for authenticated access
        dataset_id (str): The Pennsieve dataset ID (e.g. "e7d1bfbe-2899-4f58-afe9-214cad4ad46a")

    Returns:
        str: The package ID of the .bdf file

    Raises:
        Exception: If no .bdf file is found or if multiple are found
    '''

    # Construct URL
    url = f"https://api.pennsieve.io/datasets/{dataset_id}/packages"

    headers = {
        "Authorization": f"Bearer {session_key}",
        "Content-Type": "application/json"
    }

    # Make request
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Extract children list (packages)
    packages = data.get("packages", [])
    print(f"Found {len(packages)} packages in dataset {dataset_id}")

    # Find packages ending in .bdf (case-insensitive)
    bdf_packages = [
        {"package_id": pkg["content"]["nodeId"], "name": pkg["content"]["name"]}
        for pkg in packages
        if pkg.get("content", {}).get("name", "").lower().endswith(".bdf")
    ]
    print(f"Found {len(bdf_packages)} .bdf packages in dataset {dataset_id}")

    if len(bdf_packages) == 0:
        raise Exception(f"No .bdf package found in dataset {dataset_id}")
    elif len(bdf_packages) > 1:
        raise Exception(f"Multiple .bdf packages found in dataset {dataset_id}: {bdf_packages}")

    return bdf_packages[0]

def createAnnotationLayer(session_key, bdf_package_id):
    '''
    Creates an annotation layer for the specified BDF package.

    Parameters:
        session_key (str): Bearer token for authenticated access
        bdf_package_id (str): The package ID of the BDF file

    Returns:
        bool: True if the annotation layer was created successfully, False otherwise
    '''

    from datetime import datetime
    now = datetime.now()

    url = f"https://api.pennsieve.io/timeseries/{bdf_package_id}/layers"
    params = {
        "name": f"Imported Annotations {now}",
        "description": "Annotations autoimported by `https://github.com/Pennsieve/bdf-annotations-extractor`",
    }
    headers = {
        "Authorization": f"Bearer {session_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=params, headers=headers)
    
    return json.loads(response.text)

def getChannels(session_key, timeSeriesPackageID):
    '''
    Retrieves the channels for the specified BDF package.

    Parameters:
        session_key (str): Bearer token for authenticated access
        timeSeriesPackageID (str): The package ID of the BDF file

    Returns:
        list: A list of channels in the BDF package
    '''


    url = f"https://api.pennsieve.io/timeseries/{timeSeriesPackageID}/channels"
    headers = {
        "Authorization": f"Bearer {session_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    channels = json.loads(response.text)
    channel_ids = [entry["content"]["id"] for entry in channels]
    
    return channel_ids

def createAnnotation(session_key,channels,timeseriesId,timeseriesIdPackageName,annotations):
    '''
    Creates an annotation in the specified timeseries layer.

    Parameters:
        session_key (str): Bearer token for authenticated access
        timeseries_id (str): The ID of the timeseries layer
        name (str): The name of the annotation
        label (str): The label of the annotation
        start (int): Start time in milliseconds
        end (int): End time in milliseconds
        relative_start (float): Relative start time as a fraction of the total duration
        relative_end (float): Relative end time as a fraction of the total duration

    Returns:
        dict: The created annotation object
    '''

    url = f"https://api.pennsieve.io/timeseries/{timeseriesIdPackageName}/layers/{timeseriesId}/annotations?startAtEpoch=true"
    headers = {
        "Authorization": f"Bearer {session_key}",
        "Content-Type": "application/json"
    }

    for annotation in annotations:
        params = {
            "name": annotation["name"],
            "label": annotation["label"],
            "start": int(annotation["relative_start"]),
            "end": (int(annotation["relative_end"])) + 10000,
            "layer_id": timeseriesId,
            "channelIds": channels,
        } 
        print(f"Creating annotation: {params}")
        response = requests.post(url, json=params, headers=headers)
        response.raise_for_status()

        if response.status_code != 201:
            print(f"Failed to create annotation: {response.text}")
            continue
    
    return json.loads(response.text)