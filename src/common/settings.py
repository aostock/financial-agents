import base64
from langchain_core.runnables import RunnableConfig
import json


class Settings():
    def __init__(self, config: RunnableConfig):
        self.config = config
        self.dict = self.get_settings(config)


    def get_settings(self, config: RunnableConfig) -> dict:
        settings = config.get("configurable", {}).get("x-settings", "")
        if settings == "":
            return None
        # base64 decode
        settings = base64.b64decode(settings).decode('utf-8')
        return json.loads(settings)

    def get_intent_recognition_model(self) -> dict:
        return self.dict.get("intentRecognitionModel", {})
    
    def get_analysis_model(self) -> dict:
        return self.dict.get("analysisModel", {})

    def get_model_list(self) -> list:
        intent_recognition_model = self.get_intent_recognition_model()
        analysis_model = self.get_analysis_model()
        return [
            {
                "model_name": intent_recognition_model.get("model", ""),
                "litellm_params": intent_recognition_model,
            },
            {
                "model_name": analysis_model.get("model", ""),
                "litellm_params": analysis_model,
            }
        ]

    def get_remote_financial_data_api_url(self) -> str:
        return self.dict.get("remoteFinancialDataApiUrl", "")
    
    def get_remote_financial_data_api_key(self) -> str:
        return self.dict.get("remoteFinancialDataApiKey", "")
