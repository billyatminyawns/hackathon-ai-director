from typing import Optional

import requests
from pydantic import BaseModel

URL = "https://api.staging.wellsaidlabs.com/v1/tts"
headers = {
    "x-api-key": "3dc733a2-874f-4a0e-b239-201c06dc89f2",
}


class Avatar(BaseModel):
    name: str
    id: int
    style: Optional[str] = None
    gender: Optional[str] = None
    accent_type: Optional[str] = None
    characteristics: Optional[list[str]] = None
    otherTags: Optional[list[str]] = None
    preview_audio: Optional[str] = None
    locale: Optional[str] = None
    language: Optional[str] = None
    language_variant: Optional[str] = None
    source: Optional[str] = None


class AvatarsContent(BaseModel):
    avatars: list[Avatar]


class AvatarCharacteristics(BaseModel):
    characteristics: set[str]


class AvatarCriterion(BaseModel):
    name: str
    options: list[str]


class AvatarCriteria(BaseModel):
    criteria: list[AvatarCriterion]


def get_all_avatars():
    response = requests.get(f"{URL}/avatars", headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"Error from WellSaid API: {response.text}")

    avatars: list[Avatar] = [Avatar(**item) for item in response.json().get("avatars")]
    return [a for a in avatars if a.language == "English"]


AVATARS_EN = get_all_avatars()
AVATAR_TO_ID = {f"{a.name} {a.style} ({a.language_variant})": a.id for a in AVATARS_EN}
AVATAR_TO_CHARACTERISTICS = {
    f"{a.name} {a.style} ({a.language_variant})": f"{a.gender}, {a.characteristics}"
    for a in AVATARS_EN
}
