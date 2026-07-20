#include "dock_sway/DockSway.hh"

#include <chrono>
#include <gz/plugin/Register.hh>
#include <gz/sim/Util.hh>
#include <gz/common/Console.hh>

namespace dock_sway {
void DockSway::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &eventMgr
) {
    this->_model = gz::sim::Model(entity);
    if (!this->_model.Valid(ecm)) {
        gzerr << "DockSway: not attached to a model, disabling.\n";
        return;
    }

    this->_params.enabled = sdf->Get<bool>("enabled", true).first;
    this->_params.period = sdf->Get<double>("period", 4.0).first;
    this->_params.heave_amplitude = sdf->Get<double>("heave_amplitude", 0.1).first;
    this->_params.sway_amplitude = sdf->Get<double>("sway_amplitude", 0.1).first;
    this->_params.heave_phase = sdf->Get<double>("heave_phase", 0.0).first;
    this->_params.sway_phase = sdf->Get<double>("sway_phase", 1.5708).first;

    this->_home_pose = gz::sim::worldPose(entity, ecm);
    this->_home_yaw = this->_home_pose.Rot().Yaw();

    this->_valid = true;
    gzmsg << "DockSway: configured, period " << this->_params.period << " s.\n";
}

void DockSway::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm
) {
    if (!this->_valid || info.paused) {
        return;
    }

    const double t = std::chrono::duration<double>(info.simTime).count();

    auto offset = WorldOffset(t, this->_params, this->_home_yaw);

    auto target = gz::math::Pose3d(this->_home_pose.Pos() + offset, this->_home_pose.Rot());

    this->_model.SetWorldPoseCmd(ecm, target);
}
} // namespace dock_sway

GZ_ADD_PLUGIN(
    dock_sway::DockSway,
    gz::sim::System,
    dock_sway::DockSway::ISystemConfigure,
    dock_sway::DockSway::ISystemPreUpdate
)

GZ_ADD_PLUGIN_ALIAS(dock_sway::DockSway, "dock::DockSway")
