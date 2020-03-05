# Model

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The identifier of this entity | 
**name** | **str** | The name of the object | 
**description** | **str** | The description of the object | [optional] 
**model_kind** | **str** | The kind of this model (e.g. acoustic, language ...) | 
**text** | **str** | The text used to adapt this language model | [optional] 
**base_model** | [**Model**](Model.md) | The base model used for adaptation | [optional] 
**datasets** | [**list[Dataset]**](Dataset.md) | Datasets used for adaptation | [optional] 
**properties** | **dict(str, str)** | The custom properties of this entity | [optional] 
**locale** | **str** | The locale of the contained data | 
**last_action_date_time** | **datetime** | The time-stamp when the current status was entered | [optional] 
**status** | **str** | The status of the object | [optional] 
**created_date_time** | **datetime** | The time-stamp when the object was created | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


