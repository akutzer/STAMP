try:
    from huggingface_hub import login
    from transformers import AutoModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conchv1_5 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[conch1_5]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def conch1_5() -> Extractor:
    login(new_session=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    model, eval_transform = titan.return_conch()
    return Extractor(
        model=model, transform=eval_transform, identifier="mahmood-conch1_5"
    )
