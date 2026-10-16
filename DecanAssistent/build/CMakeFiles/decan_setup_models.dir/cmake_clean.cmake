file(REMOVE_RECURSE
  "CMakeFiles/decan_setup_models"
  "models/coat_of_arms_detector.onnx"
  "models/ocr_handwriting.onnx"
  "models/ocr_printed.onnx"
  "models/seal_stamp_detector.onnx"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/decan_setup_models.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
