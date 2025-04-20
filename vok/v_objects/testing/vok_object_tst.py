import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

# import the fixed class
from dawn_vok.vok.v_objects.vok_object import VOKObject
# Dummy subclass for testing
class DummyVOK(VOKObject):
    @classmethod
    def get_db_name(cls):
        return "dummy_db"

    @classmethod
    def get_collection_name(cls):
        return "dummy_coll"

# Fixtures to patch IDUtils and DictUtils
@pytest.fixture(autouse=True)
def patch_idutils(monkeypatch):
    import dawn_vok.utils.id_utils as id_utils_module
    monkeypatch.setattr(
        id_utils_module.IDUtils,
        'get_system_unique_id',
        lambda x: f"sys_{x['uid']}"
    )
    monkeypatch.setattr(
        id_utils_module.IDUtils,
        'get_id',
        lambda parts: "_".join(parts)
    )

@pytest.fixture(autouse=True)
def patch_dictutils(monkeypatch):
    import dawn_vok.utils.dict_utils as dict_utils_module
    def put_datetime(d, key, value):
        d[key] = value.isoformat()
    monkeypatch.setattr(
        dict_utils_module.DictUtils,
        'put_datetime',
        put_datetime
    )

# Tests

def test_init_and_system_uid():
    obj = DummyVOK(uid="123", obj_type="test_type", name="test_name")
    assert obj.uid == "123"
    assert obj.obj_type == "test_type"
    assert obj.system_uid == "sys_123"
    assert obj.name == "test_name"
    assert isinstance(obj.updated_at, datetime)


def test_to_dict():
    obj = DummyVOK(uid="123", obj_type="test_type", name="test_name")
    d = obj.to_dict()
    assert d['_id'] == 'test_type_123'
    assert d['obj_type'] == 'test_type'
    assert d['uid'] == '123'
    assert d['system_uid'] == 'sys_123'
    assert d['name'] == 'test_name'
    assert 'updated_at' in d


def test_populate_from_dict():
    sample_time = datetime(2020, 1, 1)
    data = {
        'obj_type': 'new_type',
        'uid': '456',
        'system_uid': 'sys_456',
        'meta_data': {'a': 1},
        'name': 'new_name',
        'updated_at': sample_time.isoformat()
    }
    obj = DummyVOK(uid="123", obj_type="test_type")
    obj.populate_from_dict(data)
    assert obj.obj_type == 'new_type'
    assert obj.uid == '456'
    assert obj.system_uid == 'sys_456'
    assert obj.meta_data == {'a': 1}
    assert obj.name == 'new_name'
    assert obj.updated_at == sample_time


def test_save_and_load_db(monkeypatch):
    fake_coll = MagicMock()
    monkeypatch.setattr(
        'dawn_vok.db.mongo_utils.MongoUtils.get_collection',
        lambda db, coll: fake_coll
    )
    obj = DummyVOK(uid="789", obj_type="test_type")
    # save by uid
    obj.save_to_db(use_system_uid=False)
    fake_coll.update_one.assert_called_once_with(
        {'uid': '789'}, {'$set': obj.to_dict()}, upsert=True
    )
    fake_coll.reset_mock()
    # save by system_uid
    obj.save_to_db(use_system_uid=True)
    fake_coll.update_one.assert_called_once_with(
        {'system_uid': 'sys_789'}, {'$set': obj.to_dict()}, upsert=True
    )
    # prepare load
    saved = obj.to_dict()
    fake_coll.find_one.return_value = saved
    loaded = DummyVOK(uid="789", obj_type="test_type")
    loaded.load_from_db(use_system_uid=False)
    assert loaded.uid == obj.uid
    assert loaded.obj_type == obj.obj_type
    assert loaded.system_uid == obj.system_uid


def test_class_methods_get_by(monkeypatch):
    fake_coll = MagicMock()
    monkeypatch.setattr(
        'dawn_vok.db.mongo_utils.MongoUtils.get_collection',
        lambda db, coll: fake_coll
    )
    # sample data
    data = {
        '_id': 'test_type_111',
        'obj_type': 'test_type',
        'uid': '111',
        'system_uid': 'sys_111',
        'meta_data': {},
        'name': 'n',
        'updated_at': datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    }
    fake_coll.find_one.return_value = data
    # get_by_uid
    obj = DummyVOK.get_by_uid('111', populate=True)
    assert isinstance(obj, DummyVOK)
    assert obj.uid == '111'
    obj_dict = DummyVOK.get_by_uid('111', populate=False)
    assert isinstance(obj_dict, dict)
    # get_by_system_uid
    obj2 = DummyVOK.get_by_system_uid('sys_111', populate=True)
    assert isinstance(obj2, DummyVOK)
    obj_dict2 = DummyVOK.get_by_system_uid('sys_111', populate=False)
    assert isinstance(obj_dict2, dict)
