import torchaudio
from torch.utils.data import Dataset
from data.parameters import PATH
from data.transforms import remove_elements, process_labels, resample, add_noise, shift_pitch, stretch_time, mel_spectrogram

_root_path_for_data = PATH

class dataset_wrapper(
    Dataset
    ):
    
    def __init__(
        self,
        speech_dataset,
        transforms
        ):
      
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.speech_dataset = speech_dataset

    def __getitem__(
        self, 
        item
        ):
      
        element = self.speech_dataset[item]
        for t in self.transforms:
            element = t(element)
        return element

    def __len__(
        self
        ):
      
        return len(self.speech_dataset)

def build_datasets(
    ):
    
    train_speech_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=_root_path_for_data, download=True, subset="training")
    train_speech_dataset = dataset_wrapper(
        speech_dataset=train_speech_dataset,
        transforms=[
            remove_elements(),
            process_labels(),
            shift_pitch(),
            stretch_time(),
            add_noise(),
            resample(),
            mel_spectrogram()
        ],
    )
    validation_speech_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=_root_path_for_data, download=True, subset="validation")
    validation_speech_dataset = dataset_wrapper(
        speech_dataset=validation_speech_dataset,
        transforms=[
            remove_elements(),
            process_labels(),
            resample(),
            mel_spectrogram()
        ],
    )
    return train_speech_dataset, validation_speech_dataset