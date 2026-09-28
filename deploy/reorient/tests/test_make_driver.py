# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import pytest
from wuji_reorient_deploy import drivers as drivers_mod
from wuji_reorient_deploy.constants import load_constants
from wuji_reorient_deploy.drivers import MockHandDriver, make_driver


def test_mock_short_circuits_to_mock_driver():
  c = load_constants(2, "right")
  d = make_driver(2, "right", mock=True, constants=c)
  assert isinstance(d, MockHandDriver)


@pytest.mark.parametrize(
  "gen,expected_attr", [(1, "WujiHand1Driver"), (2, "WujiHand2Driver")]
)
def test_gen_routes_to_correct_real_driver(gen, expected_attr, monkeypatch):
  built = {}

  def _fake_factory(name):
    def _ctor(constants, hand_side, serial_number, **kw):
      built["name"] = name
      built["hand_side"] = hand_side
      return object()

    return _ctor

  monkeypatch.setattr(drivers_mod, "WujiHand1Driver", _fake_factory("WujiHand1Driver"))
  monkeypatch.setattr(drivers_mod, "WujiHand2Driver", _fake_factory("WujiHand2Driver"))

  c = load_constants(gen, "right")
  make_driver(gen, "right", serial_number="", constants=c)
  assert built["name"] == expected_attr


def test_invalid_gen_raises_value_error():
  c = load_constants(2, "right")
  with pytest.raises(ValueError):
    make_driver(3, "right", constants=c)
