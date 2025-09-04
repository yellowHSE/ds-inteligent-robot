cmake_minimum_required(VERSION 3.8)
project(parking_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 필수 패키지 로드
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 서비스 파일 목록
set(srv_files
   "srv/CarInfo.srv"
   "srv/CarClass.srv"
   "srv/Entry.srv"
   "srv/CarSize.srv"
   "srv/Exit.srv"
   "srv/DetectPerson.srv"
)

# 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
)

# 런타임 의존성 export
ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
