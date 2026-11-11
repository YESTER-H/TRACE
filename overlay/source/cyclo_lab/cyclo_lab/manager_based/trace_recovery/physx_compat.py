# SPDX-License-Identifier: Apache-2.0
"""Exact PhysX contact reporting for the new importer's nested rigid links.

No URDF, importer or third-party implementation is changed. Native PhysX
ContactSensor currently assumes siblings when sensing multiple bodies; one
native sensor per explicitly named rigid link avoids that assumption.
"""
import xml.etree.ElementTree as ET
from .body_adapter import BODY_NAMES


def spawn_reported_urdf(prim_path, cfg, translation=None, orientation=None):
    from isaaclab.sim import spawn_from_urdf
    from pxr import Usd, UsdPhysics, PhysxSchema
    root = spawn_from_urdf(prim_path,cfg,translation,orientation)
    names = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr(0.)
            names.append(prim.GetName())
    if sorted(names) != sorted(BODY_NAMES):
        raise RuntimeError(f"Expected exactly 25 named contact-reporting rigid links: {names}")
    return root


def configure_link_sensors(scene, urdf_path):
    from isaaclab.sensors import ContactSensorCfg
    document = ET.parse(urdf_path).getroot()
    parents = {j.find("child").get("link"):j.find("parent").get("link") for j in document.findall("joint")}
    def path(name):
        return path(parents[name])+"/"+name if name in parents else name
    scene.contact_forces = None
    scene.head_contact_forces = None
    for i,name in enumerate(BODY_NAMES):
        setattr(scene,f"logical_contact_{i:02d}",ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/"+path(name),
            history_length=3,track_air_time=False,debug_vis=False,
        ))
