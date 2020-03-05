# Test

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results_url** | **str** | The URL that can be used to download the test results.  Each line in the file represents a tab separated list of filename, reference transcription and decoder output.                The URL will only be valid, if the test completed successfully | [optional] 
**id** | **str** | The identifier of this entity | 
**word_error_rate** | **float** | The word error rate of the tested model | [optional] 
**last_action_date_time** | **datetime** | The time-stamp when the current status was entered | [optional] 
**status** | **str** | The status of the object | [optional] 
**created_date_time** | **datetime** | The time-stamp when the object was created | [optional] 
**models** | [**list[Model]**](Model.md) | Information about the models used for this accuracy test | 
**dataset** | [**Dataset**](Dataset.md) | Information about the dataset used in the test | [optional] 
**name** | **str** | The name of the object | 
**description** | **str** | The description of the object | [optional] 
**properties** | **dict(str, str)** | The custom properties of this entity | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


