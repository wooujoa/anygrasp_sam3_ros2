# anygrasp_sam3_ros2

SAM3가 만든 point cloud 토픽(`/yolo/target_pc` 또는 `/yolo/object_pc`)을 받아서 AnyGrasp SDK로 grasp를 추론하고, RViz2에서 바로 확인할 수 있도록 `PoseArray`, `PoseStamped`, `MarkerArray`를 발행하는 ROS 2 패키지입니다.

## 발행 토픽
- `/anygrasp/grasps` (`geometry_msgs/PoseArray`)
- `/anygrasp/best_grasp` (`geometry_msgs/PoseStamped`)
- `/anygrasp/best_width` (`std_msgs/Float32`)
- `/anygrasp/grasp_markers` (`visualization_msgs/MarkerArray`)

## 기본 입력 토픽
- `/yolo/target_pc`
- `/yolo/object_pc`

## 워크스페이스에 넣기
```bash
cd ~/colcon_ws/src
cp -r /path/to/anygrasp_sam3_ros2 .
```

## 빌드
```bash
cd ~/colcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select anygrasp_sam3_ros2
source ~/colcon_ws/install/setup.bash
```

## 실행 전
AnyGrasp SDK 쪽은 라이선스 등록과 체크포인트가 준비되어 있어야 합니다. 공식 README에는 Python 3.6-3.13, PyTorch 1.7.1+, MinkowskiEngine v0.5.4가 요구되고, `demo.py` 예시에서는 `AnyGrasp(cfgs)`를 생성한 뒤 `load_net()` 후 `get_grasp(points, colors, lims=...)`를 호출합니다. 또한 2024-05 업데이트 기준으로 `dense_grasp`, `apply_object_mask`, `collision_detection` 플래그가 제공됩니다. citeturn142826view1turn142826view0

## 실행
```bash
conda activate anygrasp
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash

ros2 launch anygrasp_sam3_ros2 anygrasp_from_sam3.launch.py
```

또는:
```bash
ros2 run anygrasp_sam3_ros2 anygrasp_from_topic_node \
  --ros-args \
  -p sdk_root:=/home/jwg/anygrasp_sdk/grasp_detection \
  -p checkpoint_path:=/home/jwg/anygrasp_sdk/ckpt/checkpoint_detection.tar \
  -p target_cloud_topic:=/yolo/target_pc \
  -p use_object_cloud_as_input:=false
```

## RViz2
RViz2에서 아래 항목 추가:
- `PointCloud2` : `/yolo/target_pc`
- `PoseArray` : `/anygrasp/grasps`
- `MarkerArray` : `/anygrasp/grasp_markers`
- `Pose` : `/anygrasp/best_grasp`

고정 프레임은 입력 point cloud의 frame_id와 같게 두면 됩니다.

## 설계 포인트
- 기존 SAM3 one-shot 노드는 그대로 두고, 그 결과 토픽만 소비합니다.
- 입력 cloud에 RGB가 없더라도 동작하도록 회색 색상을 채워 AnyGrasp에 전달합니다.
- 입력 cloud bounds로부터 자동으로 `lims`를 계산합니다.
- RViz2에서 보기 쉽게 각 grasp를 palm/finger 3개 cube marker로 근사해 그립니다.

## 참고
Contact-GraspNet은 논문에서 전체 장면 또는 대상 주변의 local ROI를 입력으로 쓰고, 대상 3D centroid 주변에 가장 긴 변의 2배, 최소 0.3m, 최대 0.6m 크기 cube ROI를 쓰는 구성을 제안합니다. 네가 쓰던 SAM3 노드도 비슷하게 target/object/background cloud를 따로 발행하고 있어, 이번 AnyGrasp 브리지는 그 구조를 그대로 활용하는 방식입니다. fileciteturn1file7 fileciteturn0file0

AnyGrasp 논문/SDK 쪽도 조밀한 7-DoF grasp 예측, 실시간 수준의 추론, 충돌/객체성 기반 필터링을 강조하고 있고, SDK README는 `dense_grasp`, `apply_object_mask`, `collision_detection` 플래그를 공식 지원한다고 명시합니다. fileciteturn1file10 citeturn142826view1turn142826view0
