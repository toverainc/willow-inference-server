import logging
import torch
from ts.torch_handler.base_handler import BaseHandler
import base64
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from transformers import (
    VisionTextDualEncoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)
logger = logging.getLogger(__name__)
torch.set_num_threads(1)

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BILINEAR),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    #x is Image
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

class MMFHandler(BaseHandler):
    """
    Transformers handler class for  MMFTransformerWithVideoAudio model.
    """
    def __init__(self):
        super(MMFHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        #load model
        self.model = VisionTextDualEncoderModel.from_pretrained(model_dir)
        for param in self.model.parameters(): #for loop freezes model
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()
        logger.debug('model from path {0} loaded successfully'.format(model_dir))
        #load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('.')
        #create image_processing
        feature_extractor = AutoFeatureExtractor.from_pretrained('.')
        image_transformations = Transform(
            self.model.config.vision_config.image_size,
            feature_extractor.image_mean,
            feature_extractor.image_std
        )
        image_transformations = torch.jit.script(image_transformations)
        self.image_processing = image_transformations

        self.initialized = True

    def preprocess(self, requests):
        modes = []
        images = []
        texts = []
        for data in requests:
            text = data.get('text')
            if text:
                text = text.decode('utf-8')
                modes.append('text')
                texts.append(text)
            else: #assume image
                image = data.get("data") or data.get("body")
                image = decode_image(torch.from_numpy(np.frombuffer(image, dtype=np.uint8)), mode=ImageReadMode.RGB)
                image = self.image_processing(image)
                modes.append('image')
                images.append(image)

        if texts:
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)

        return {
            'images': torch.stack(images).to(self.device) if images else None,
            'texts': {'input_ids':input_ids, 'attention_mask':attention_mask} if texts else None,
            'modes':modes,
        }

    def inference(self, data):
        if data['images'] is not None:
            image_features = list(self.model.get_image_features(pixel_values=data['images']).to('cpu').numpy())
        if data['texts'] is not None:
            text_features = list(self.model.get_text_features(**data['texts']).to('cpu').numpy())
        res = []
        for mode in data['modes']:
            if mode == "image":
                res.append(image_features.pop(0))
            else: #text
                res.append(text_features.pop(0))
        return np.stack(res)

_service = MMFHandler()
def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e