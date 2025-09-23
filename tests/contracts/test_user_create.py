# tests/contracts/test_user_create.py
import pytest
from hypothesis import given, strategies as st
from contracts.user.create.validators import pre_create, post_create
from contracts.user.create.create import create

# Hypothesis strategies
def user_in_strategy():
    return st.fixed_dictionaries({
        "email": st.emails(),
        "name": st.text(min_size=1, max_size=80)
    })

@given(data=user_in_strategy(), referrer=st.none() | st.text(min_size=1, max_size=64))
def test_create_contract(data, referrer):
    pre_create(data, referrer)
    out = create(data, referrer)
    post_create({"data": data, "referrer": referrer}, out)
