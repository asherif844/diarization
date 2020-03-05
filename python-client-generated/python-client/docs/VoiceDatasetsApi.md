# swagger_client.VoiceDatasetsApi

All URIs are relative to *https://eastus2.cris.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**delete_voice_dataset**](VoiceDatasetsApi.md#delete_voice_dataset) | **DELETE** /api/texttospeech/v2.0/datasets/{id} | Deletes the voice dataset with the given id.
[**get_supported_locales_for_voice_datasets**](VoiceDatasetsApi.md#get_supported_locales_for_voice_datasets) | **GET** /api/texttospeech/v2.0/datasets/locales | Gets a list of supported locales for custom voice data imports.
[**get_voice_datasets**](VoiceDatasetsApi.md#get_voice_datasets) | **GET** /api/texttospeech/v2.0/datasets | Gets all voice datasets.
[**update_voice_dataset**](VoiceDatasetsApi.md#update_voice_dataset) | **PATCH** /api/texttospeech/v2.0/datasets/{id} | Updates the mutable details of the voice dataset identified by its ID.
[**upload_voice_dataset**](VoiceDatasetsApi.md#upload_voice_dataset) | **POST** /api/texttospeech/v2.0/datasets/upload | Uploads data and creates a new voice data object.


# **delete_voice_dataset**
> delete_voice_dataset(id)

Deletes the voice dataset with the given id.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: subscription_key
configuration = swagger_client.Configuration()
configuration.api_key['Ocp-Apim-Subscription-Key'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Ocp-Apim-Subscription-Key'] = 'Bearer'
# Configure API key authorization: token
configuration = swagger_client.Configuration()
configuration.api_key['Authorization'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.VoiceDatasetsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The identifier of the voice dataset.

try:
    # Deletes the voice dataset with the given id.
    api_instance.delete_voice_dataset(id)
except ApiException as e:
    print("Exception when calling VoiceDatasetsApi->delete_voice_dataset: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The identifier of the voice dataset. | 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_supported_locales_for_voice_datasets**
> list[str] get_supported_locales_for_voice_datasets()

Gets a list of supported locales for custom voice data imports.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: subscription_key
configuration = swagger_client.Configuration()
configuration.api_key['Ocp-Apim-Subscription-Key'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Ocp-Apim-Subscription-Key'] = 'Bearer'
# Configure API key authorization: token
configuration = swagger_client.Configuration()
configuration.api_key['Authorization'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.VoiceDatasetsApi(swagger_client.ApiClient(configuration))

try:
    # Gets a list of supported locales for custom voice data imports.
    api_response = api_instance.get_supported_locales_for_voice_datasets()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceDatasetsApi->get_supported_locales_for_voice_datasets: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**list[str]**

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_voice_datasets**
> list[Dataset] get_voice_datasets()

Gets all voice datasets.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: subscription_key
configuration = swagger_client.Configuration()
configuration.api_key['Ocp-Apim-Subscription-Key'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Ocp-Apim-Subscription-Key'] = 'Bearer'
# Configure API key authorization: token
configuration = swagger_client.Configuration()
configuration.api_key['Authorization'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.VoiceDatasetsApi(swagger_client.ApiClient(configuration))

try:
    # Gets all voice datasets.
    api_response = api_instance.get_voice_datasets()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceDatasetsApi->get_voice_datasets: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[Dataset]**](Dataset.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_voice_dataset**
> Dataset update_voice_dataset(id, dataset_update)

Updates the mutable details of the voice dataset identified by its ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: subscription_key
configuration = swagger_client.Configuration()
configuration.api_key['Ocp-Apim-Subscription-Key'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Ocp-Apim-Subscription-Key'] = 'Bearer'
# Configure API key authorization: token
configuration = swagger_client.Configuration()
configuration.api_key['Authorization'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.VoiceDatasetsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The identifier of the voice dataset.
dataset_update = swagger_client.DatasetUpdate() # DatasetUpdate | The updated values for the voice dataset.

try:
    # Updates the mutable details of the voice dataset identified by its ID.
    api_response = api_instance.update_voice_dataset(id, dataset_update)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceDatasetsApi->update_voice_dataset: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The identifier of the voice dataset. | 
 **dataset_update** | [**DatasetUpdate**](DatasetUpdate.md)| The updated values for the voice dataset. | 

### Return type

[**Dataset**](Dataset.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **upload_voice_dataset**
> upload_voice_dataset(project_id=project_id, name=name, description=description, locale=locale, data_import_kind=data_import_kind, properties=properties, audiodata=audiodata, transcriptions=transcriptions)

Uploads data and creates a new voice data object.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: subscription_key
configuration = swagger_client.Configuration()
configuration.api_key['Ocp-Apim-Subscription-Key'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Ocp-Apim-Subscription-Key'] = 'Bearer'
# Configure API key authorization: token
configuration = swagger_client.Configuration()
configuration.api_key['Authorization'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['Authorization'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.VoiceDatasetsApi(swagger_client.ApiClient(configuration))
project_id = 'project_id_example' # str | The optional string representation of a project ID. If set, the dataset will be associated with that project. (optional)
name = 'name_example' # str | The name of this data import (always add this string for any import). (optional)
description = 'description_example' # str | Optional description of this data import. (optional)
locale = 'locale_example' # str | The locale of this data import (always add this string for any import). (optional)
data_import_kind = 'data_import_kind_example' # str | The kind of the data import (always add this string for any import). (optional)
properties = 'properties_example' # str | Optional properties of this data import (json serialized object with key/values, where all values must be strings) (optional)
audiodata = '/path/to/file.txt' # file | A zip file containing the audio data. (optional)
transcriptions = '/path/to/file.txt' # file | The transcriptions text file of the audio data. (optional)

try:
    # Uploads data and creates a new voice data object.
    api_instance.upload_voice_dataset(project_id=project_id, name=name, description=description, locale=locale, data_import_kind=data_import_kind, properties=properties, audiodata=audiodata, transcriptions=transcriptions)
except ApiException as e:
    print("Exception when calling VoiceDatasetsApi->upload_voice_dataset: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **project_id** | **str**| The optional string representation of a project ID. If set, the dataset will be associated with that project. | [optional] 
 **name** | **str**| The name of this data import (always add this string for any import). | [optional] 
 **description** | **str**| Optional description of this data import. | [optional] 
 **locale** | **str**| The locale of this data import (always add this string for any import). | [optional] 
 **data_import_kind** | **str**| The kind of the data import (always add this string for any import). | [optional] 
 **properties** | **str**| Optional properties of this data import (json serialized object with key/values, where all values must be strings) | [optional] 
 **audiodata** | **file**| A zip file containing the audio data. | [optional] 
 **transcriptions** | **file**| The transcriptions text file of the audio data. | [optional] 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

