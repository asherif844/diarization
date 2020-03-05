# swagger_client.VoiceEndpointsApi

All URIs are relative to *https://eastus2.cris.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_voice_deployment**](VoiceEndpointsApi.md#create_voice_deployment) | **POST** /api/texttospeech/v2.0/endpoints | Creates a new voice endpoint object.
[**delete_deployment**](VoiceEndpointsApi.md#delete_deployment) | **DELETE** /api/texttospeech/v2.0/endpoints/{id} | Delete the specified voice endpoint.
[**get_supported_locales_for_voice_endpoints**](VoiceEndpointsApi.md#get_supported_locales_for_voice_endpoints) | **GET** /api/texttospeech/v2.0/endpoints/locales | Gets a list of supported locales for custom voice endpoints.
[**get_voice_deployment**](VoiceEndpointsApi.md#get_voice_deployment) | **GET** /api/texttospeech/v2.0/endpoints/{id} | Gets the details of a custom voice endpoint.
[**get_voice_deployments**](VoiceEndpointsApi.md#get_voice_deployments) | **GET** /api/texttospeech/v2.0/endpoints | Gets a list of voice endpoint details.
[**update_voice_endpoint**](VoiceEndpointsApi.md#update_voice_endpoint) | **PATCH** /api/texttospeech/v2.0/endpoints/{id} | Updates the name and description of the endpoint identified by the given ID.


# **create_voice_deployment**
> create_voice_deployment(endpoint)

Creates a new voice endpoint object.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))
endpoint = swagger_client.EndpointDefinition() # EndpointDefinition | 

try:
    # Creates a new voice endpoint object.
    api_instance.create_voice_deployment(endpoint)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->create_voice_deployment: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **endpoint** | [**EndpointDefinition**](EndpointDefinition.md)|  | 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_deployment**
> delete_deployment(id)

Delete the specified voice endpoint.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The id of voice endpoint.

try:
    # Delete the specified voice endpoint.
    api_instance.delete_deployment(id)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->delete_deployment: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The id of voice endpoint. | 

### Return type

void (empty response body)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_supported_locales_for_voice_endpoints**
> list[str] get_supported_locales_for_voice_endpoints()

Gets a list of supported locales for custom voice endpoints.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))

try:
    # Gets a list of supported locales for custom voice endpoints.
    api_response = api_instance.get_supported_locales_for_voice_endpoints()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->get_supported_locales_for_voice_endpoints: %s\n" % e)
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

# **get_voice_deployment**
> Endpoint get_voice_deployment(id)

Gets the details of a custom voice endpoint.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | 

try:
    # Gets the details of a custom voice endpoint.
    api_response = api_instance.get_voice_deployment(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->get_voice_deployment: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)|  | 

### Return type

[**Endpoint**](Endpoint.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_voice_deployments**
> list[Endpoint] get_voice_deployments()

Gets a list of voice endpoint details.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))

try:
    # Gets a list of voice endpoint details.
    api_response = api_instance.get_voice_deployments()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->get_voice_deployments: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[Endpoint]**](Endpoint.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_voice_endpoint**
> Endpoint update_voice_endpoint(id, endpoint_update)

Updates the name and description of the endpoint identified by the given ID.

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
api_instance = swagger_client.VoiceEndpointsApi(swagger_client.ApiClient(configuration))
id = 'id_example' # str | The identifier of the endpoint.
endpoint_update = swagger_client.EndpointMetadataUpdate() # EndpointMetadataUpdate | The updated values for the endpoint.

try:
    # Updates the name and description of the endpoint identified by the given ID.
    api_response = api_instance.update_voice_endpoint(id, endpoint_update)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling VoiceEndpointsApi->update_voice_endpoint: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**str**](.md)| The identifier of the endpoint. | 
 **endpoint_update** | [**EndpointMetadataUpdate**](EndpointMetadataUpdate.md)| The updated values for the endpoint. | 

### Return type

[**Endpoint**](Endpoint.md)

### Authorization

[subscription_key](../README.md#subscription_key), [token](../README.md#token)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

