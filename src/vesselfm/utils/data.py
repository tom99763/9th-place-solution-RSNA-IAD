import logging

from monai import transforms
from monai.transforms import Compose

logger = logging.getLogger(__name__)


def generate_transforms(
    transforms_config: list[dict],
) -> list[transforms.Transform]:
    """
    Generate a list of transforms from a list of transform configurations.

    Args:
        transforms_config (list[dict]): List of transform configurations.

    Returns:
        list: List of transforms.
    """

    transform_list = []
    logger.debug(f"Generating {len(transforms_config)} transforms")

    for transform_config in transforms_config:
        transform_name = next(iter(transform_config))
        transform_kwargs = transform_config[transform_name]
        logger.debug(
            f"Generating transform {transform_name} with kwargs {transform_kwargs}"
        )
        transform: transforms.Transform = getattr(transforms, transform_name)(
            **transform_kwargs
        )  # type: ignore
        transform_list.append(transform)

    return Compose(transform_list)  # type: ignore
