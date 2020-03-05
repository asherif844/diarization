# ModelDefinition

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**model_kind** | **str** | The kind of this model (e.g. acoustic, language ...) | 
**text** | **str** | The text used to adapt this language model | [optional] 
**base_model** | [**ModelIdentity**](ModelIdentity.md) | The base model used for adaptation | [optional] 
**datasets** | [**list[DatasetIdentity]**](DatasetIdentity.md) | Datasets used for adaptation | [optional] 
**description** | **str** | The description of the object | [optional] 
**properties** | **dict(str, str)** | The custom properties of this entity | [optional] 
**locale** | **str** | The locale of the contained data | 
**name** | **str** | The name of the object | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


