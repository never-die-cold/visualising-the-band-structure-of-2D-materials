"""元素数据库测试。"""

import pytest

from simulator.elements import (
    ELEMENTS,
    Element,
    covalent_radius,
    get_element,
    get_element_by_z,
)


def test_basic_lookup():
    mo = get_element("Mo")
    assert mo.z == 42
    assert mo.name_zh == "钼"
    assert 95 < mo.mass < 96
    assert mo.color.startswith("#")


def test_lookup_case_insensitive_and_invalid():
    assert get_element("c").z == 6
    assert get_element("C").z == 6
    with pytest.raises(ValueError):
        get_element("Xx")


def test_by_z():
    assert get_element_by_z(6).symbol == "C"
    assert get_element_by_z(74).name_en == "Tungsten"
    with pytest.raises(ValueError):
        get_element_by_z(300)


def test_covalent_radius_values():
    # Cordero 2008 参考值
    assert covalent_radius("C") == pytest.approx(0.76)
    assert covalent_radius("S") == pytest.approx(1.05)
    assert covalent_radius("Mo") == pytest.approx(1.54)


def test_color_rgb_conversion():
    c = get_element("C")
    r, g, b = c.color_rgb
    assert (r, g, b) == pytest.approx((0x90 / 255, 0x90 / 255, 0x90 / 255))


def test_all_elements_have_consistent_data():
    for sym, el in ELEMENTS.items():
        assert isinstance(el, Element)
        assert el.symbol == sym
        assert el.z >= 1
        assert el.mass > 0
        assert 0.2 < el.covalent_radius < 2.5
        assert len(el.color) == 7 and el.color[0] == "#"
