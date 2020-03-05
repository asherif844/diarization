# swagger_client.VoiceModelsApi

All URIs are relative to *https://eastus2.cris.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_voice_model**](VoiceModelsApi.md#create_voice_model) | **POST** /api/texttospeech/v2.0/models | Creates a new voice model object.
[**delete_voice_model**](VoiceModelsApi.md#delete_voice_model) | **DELETE** /api/texttospeech/v2.0/models/{id} | Deletes the voice model with the given id.
[**get_supported_locales_for_voice_models**](VoiceModelsApi.md#get_supported_locales_for_voice_models) | **GET** /api/texttospeech/v2.0/models/locales | Gets a list of supported locales for custom voice models.
[**get_voice_model**](VoiceModelsApi.md#get_voice_model) | **GET** /api/texttospeech/v2.0/models/{id} | Gets specified voice model details.
[**get_voice_models**](VoiceModelsApi.md#get_voice_models) | **GET** /api/texttospeech/v2.0/models | Gets a list of voice model details.
[**update_voice_model**](VoiceModelsApi.md#update_voice_model) | **PATCH** /api/texttospeech/v2.0/models/{id} | Updates the metadata of the voice model identified by the given ID.


# **create_voice_model**
> create_voice_model(model_definition)

Creates a new voice model object.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))
model_definition = swagger_client.ModelDefinition() # ModelDefinition | 

try:
    # Creates a new voice model object.
    api_instance.create_voice_model(model_definition)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->create_voice_model: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **model_definition** | [**ModelDefinition**](ModelDefinition.md)|  | 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_voice_model**
> delete_voice_model(id)

Deletes the voice model with the given id.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The identifier of the voice model.

try:
    # Deletes the voice model with the given id.
    api_instance.delete_voice_model(id)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->delete_voice_model: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The identifier of the voice model. | 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_supported_locales_for_voice_models**
> list[str] get_supported_locales_for_voice_models()

Gets a list of supported locales for custom voice models.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))

try:
    # Gets a list of supported locales for custom voice models.
    api_response = api_instance.get_supported_locales_for_voice_models()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->get_supported_locales_for_voice_models: %s\n" % e)
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

# **get_voice_model**
> Model get_voice_model(id)

Gets specified voice model details.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | 

try:
    # Gets specified voice model details.
    api_response = api_instance.get_voice_model(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->get_voice_model: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)|  | 

### Return type

[**Model**](Model.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_voice_models**
> list[Model] get_voice_models()

Gets a list of voice model details.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))

try:
    # Gets a list of voice model details.
    api_response = api_instance.get_voice_models()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->get_voice_models: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[Model]**](Model.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_voice_model**
> Model update_voice_model(id, model_update)

Updates the metadata of the voice model identified by the given ID.

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
api_instance = swagger_client.VoiceModelsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The identifier of the voice model.
model_update = swagger_client.ModelUpdate() # ModelUpdate | The updated values for the voice model.

try:
    # Updates the metadata of the voice model identified by the given ID.
    api_response = api_instance.update_voice_model(id, model_update)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceModelsApi->update_voice_model: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The identifier of the voice model. | 
 **model_update** | [**ModelUpdate**](ModelUpdate.md)| The updated values for the voice model. | 

### Return type

[**Model**](Model.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

