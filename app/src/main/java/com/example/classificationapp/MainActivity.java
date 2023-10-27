package com.example.classificationapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.classificationapp.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    Bitmap bitmap;
    ImageView imgView;
    TextView tv;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button select = (Button)findViewById(R.id.button);
        imgView = findViewById(R.id.imageView);
        tv=findViewById(R.id.textView2);
        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }
        });

        Button predict = (Button)findViewById(R.id.button2);
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap resized=Bitmap.createScaledBitmap(bitmap,64,64,true);
                try {
                    Context context = v.getContext();
                    Model model = Model.newInstance(context);
                    TensorImage tbuffer = TensorImage.fromBitmap(resized);
                    ByteBuffer byteBuffer = tbuffer.getBuffer();
                    // Creates inputs for reference.

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 64, 64, 3}, DataType.UINT8);


                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);

                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] scores = outputFeature0.getFloatArray();
                    Log.d("output of score", scores.toString());
                    if(scores[4]==0.0){
                        tv.setText("IT IS A DOG");
                    }
                    else{
                        tv.setText("IT IS A CAT");
                    }
                    //tv.setText(String.valueOf(outputFeature0.getFloatValue(4)));


                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {

                }

            }
        });


    }
    @Override
    public void onActivityResult(int requestCode,
                                 int resultCode, Intent data) {
        if(data!=null){
            super.onActivityResult(requestCode, resultCode, data);
            imgView.setImageURI(data.getData());
            Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}