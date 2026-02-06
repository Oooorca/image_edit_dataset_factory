from image_edit_dataset_factory.utils.naming import (
    format_sample_id,
    instruction_ch_name,
    instruction_en_name,
    mask_name,
    result_image_name,
    source_image_name,
    validate_sample_id,
)


def test_sample_id_formatting() -> None:
    assert format_sample_id(1) == "00001"
    assert format_sample_id(42) == "00042"
    assert validate_sample_id("12345")
    assert not validate_sample_id("1234")


def test_file_naming_templates() -> None:
    sid = "00009"
    assert source_image_name(sid) == "00009.jpg"
    assert result_image_name(sid) == "00009_result.jpg"
    assert instruction_ch_name(sid) == "00009_CH.txt"
    assert instruction_en_name(sid) == "00009_EN.txt"
    assert mask_name(sid) == "00009_mask.png"
    assert mask_name(sid, 1) == "00009_mask-1.png"
