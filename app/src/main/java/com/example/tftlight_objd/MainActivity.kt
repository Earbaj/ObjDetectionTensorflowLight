package com.example.tftlight_objd

import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.tftlight_objd.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {


    lateinit var btnPrev: Button
    lateinit var btnPredict: Button
    lateinit var img_view: ImageView
    lateinit var result_txt: TextView
    lateinit var bitmap: Bitmap

    // Image Processor
    var imageProcess = ImageProcessor.Builder()
        .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR))
        .build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnPrev = findViewById(R.id.btn_show)
        btnPredict = findViewById(R.id.btn_predict)
        img_view = findViewById(R.id.img_view)
        result_txt = findViewById(R.id.txt_result)

        var lebels = application.assets.open("labels.txt").bufferedReader().readLines()

        // Button listener for pic image from gellary and show in image view
        btnPrev.setOnClickListener {
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent, 100)
        }

        //Button listener for predict image with labels
        btnPredict.setOnClickListener {

            var tensorImage = TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)

            tensorImage = imageProcess.process(tensorImage)

            val model = MobilenetV110224Quant.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIndx = 0

            outputFeature0.forEachIndexed { index, fl ->
                if (outputFeature0[maxIndx] < fl){
                    maxIndx = index
                }
            }

            result_txt.setText(lebels[maxIndx])

            // Releases model resources if no longer used.
            model.close()

        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // Code for get image from gallery
        if(requestCode == 100){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            img_view.setImageBitmap(bitmap)
        }
    }
}