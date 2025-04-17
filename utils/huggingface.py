import logging

logger = logging.getLogger(__name__)


def suggest_automodel_class(model_name):
    from huggingface_hub import model_info
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForImageClassification,
        AutoModelForObjectDetection,
        AutoModelForSemanticSegmentation,
        AutoModelForAudioClassification,
    )

    try:
        info = model_info(model_name)
        pipeline_tag = info.pipeline_tag

        if pipeline_tag == "feature-extraction":
            return AutoModel
        elif pipeline_tag == "fill-mask":
            return AutoModelForMaskedLM
        elif pipeline_tag == "question-answering":
            return AutoModelForQuestionAnswering
        elif pipeline_tag == "sentiment-analysis" or pipeline_tag == "text-classification":
            return AutoModelForSequenceClassification
        elif pipeline_tag == "token-classification":
            return AutoModelForTokenClassification
        elif pipeline_tag == "text-generation":
            return AutoModelForCausalLM
        elif pipeline_tag == "text2text-generation":
            return AutoModelForSeq2SeqLM
        elif pipeline_tag == "zero-shot-classification":
            return AutoModelForSequenceClassification  # Or potentially AutoModel
        elif pipeline_tag == "table-question-answering":
            return AutoModelForQuestionAnswering # May need a specific class
        elif pipeline_tag == "visual-question-answering":
            return AutoModelForQuestionAnswering # May need a specific class
        elif pipeline_tag == "image-classification":
            return AutoModelForImageClassification
        elif pipeline_tag == "object-detection":
            return AutoModelForObjectDetection
        elif pipeline_tag == "semantic-segmentation":
            return AutoModelForSemanticSegmentation
        elif pipeline_tag == "audio-classification":
            return AutoModelForAudioClassification
        elif pipeline_tag == "summarization":
            return AutoModelForSeq2SeqLM
        elif pipeline_tag == "translation":
            return AutoModelForSeq2SeqLM
        elif pipeline_tag == "text-to-speech":
            # No direct AutoModel yet, might need to use specific model class
            return None # Or suggest a base AutoModel
        else:
            # Fallback based on model name keywords (less reliable)
            model_name_lower = model_name.lower()
            if "gpt" in model_name_lower or "llama" in model_name_lower or "codegen" in model_name_lower:
                return AutoModelForCausalLM
            elif "bert" in model_name_lower or "roberta" in model_name_lower or "distilbert" in model_name_lower or "albert" in model_name_lower:
                return AutoModel
            elif "t5" in model_name_lower or "bart" in model_name_lower or "mt5" in model_name_lower:
                return AutoModelForSeq2SeqLM
            else:
                return AutoModel  # More generic fallback
    except Exception as e:
        logger.info(f"Could not retrieve model info for {model_name}: {e}")
        return AutoModel  # Fallback to a generic AutoModel
