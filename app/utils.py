import uuid
import hashlib
from bson import ObjectId

def get_uuid3(text):
    """
    Generate a UUID3 hash from the given text.
    """
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, text))


def get_object_id(text):
    hash_bytes = hashlib.md5(text.encode("utf-8")).digest()[:12]
    return ObjectId(hash_bytes)