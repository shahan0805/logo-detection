 git clone https://github.com/tensorflow/models.git
 cd models/research/object_detection
 wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
 tar zxvf ssd_inception_v2_coco_2018_01_28.tar.gz
 git clone https://github.com/satojkovic/DeepLogo.git
 cd DeepLogo
 wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
 tar zxvf flickr_logos_27_dataset.tar.gz
 cd flickr_logos_27_dataset
 tar zxvf flickr_logos_27_dataset_images.tar.gz
 cd ../
 cd DeepLogo
 python preproc_annot.py
 python gen_tfrecord.py --train_or_test train --csv_input flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation_cropped.txt --img_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path train.tfrecord
 python gen_tfrecord.py --train_or_test test --csv_input flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt --img_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path test.tfrecord
 OBJECT_DETECTION_API_DIR={path to tensorflow/models/research/object_detection}
 ln -s ${OBJECT_DETECTION_API_DIR}/ssd_inception_v2_coco_2018_01_28 ssd_inception_v2_coco_2018_01_28
 python ${OBJECT_DETECTION_API_DIR}/legacy/train.py --logtostderr --pipeline_config_path=ssd_inception_v2_coco.config --train_dir=training
 STEPS={the number of steps when the model is saved}
 python ${OBJECT_DETECTION_API_DIR}/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=ssd_inception_v2_coco.config --trained_checkpoint_prefix=model.ckpt-${STEPS} --output_directory=logos_inference_graph
 python logo_detection.py --model_name logos_inference_graph/ --label_map flickr_logos_27_label_map.pbtxt --test_annot_text flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt --test_image_dir flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_dir detect_results
 wc -l flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt
     438 flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt

eval_config: {
  num_examples: 438
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}
 python ${OBJECT_DETECTION_API_DIR}/legacy/eval.py --logtostderr --checkpoint_dir=training --eval_dir=eval --pipeline_config_path=training/pipeline.config
 tensorboard --logdir=eval/
