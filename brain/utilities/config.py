"""
Created on 01-03-2024

@authors:
* Austin Dibble, University of Glasgow

Base augmentation config
"""

from pydantic import BaseModel, Field

class AugmentConfig(BaseModel):
    """Augmentation configurations"""
    augmentation: bool = Field(False, title="use or not data augmentation")
    prob_overall: float = Field(0.9, title="overall probability to apply data augmentation (training ONLY)")

    prob_flip: float = Field(0.0, title="probability to apply 'VerticalFlip'")
    prob_inho: float = Field(1.0, title="probability to apply 'InhomogeneityNoiseAugment'")

    prob_geom: float = Field(1.0, title="probability to apply a geometric transformation (one of the following)")
    prob_grid: float = Field(0.0, title="probability to apply 'GridDistortion'")
    prob_resi: float = Field(0.0, title="probability to apply 'RandomResizedCrop'")
    prob_rota: float = Field(1.0, title="probability to apply 'RotationAugment'")
    prob_tran: float = Field(1.0, title="probability to apply 'TranslationAugment'")
    prob_moti: float = Field(1.0, title="probability to apply 'RandomMotionAugment'")

    prob_colo: float = Field(1.0, title="probability to apply a color transformation (one of the following)")
    prob_blur: float = Field(1.0, title="probability to apply 'Blur'")
    prob_down: float = Field(1.0, title="probability to apply 'Downscale'")
    prob_salt: float = Field(1.0, title="probability to apply 'SaltAndPepperNoiseAugment'")
    prob_gaus: float = Field(1.0, title="probability to apply 'GaussianNoiseAugment'")
    prob_ghos: float = Field(1.0, title="probability to apply 'GhostingAugment'")
    prob_gamm: float = Field(1.0, title="probability to apply 'GammaNoiseAugment'")
    prob_cont: float = Field(1.0, title="probability to apply 'ContrastNoiseAugment'")
    prob_slic: float = Field(1.0, title="probability to apply 'SliceSpacingNoiseAugment'")
    prob_bias: float = Field(1.0, title="probability to apply 'BiasNoiseAugment'")
    prob_neck: float = Field(0.0, title="probability to apply 'SliceRepetitionNeckNoiseAugment'")
    