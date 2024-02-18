package com.berkzerey.aidoc;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;


public class NextActivity extends AppCompatActivity {

    TextView hastalik,tedavi;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_next);
        getSupportActionBar().setDisplayShowTitleEnabled(false);

        Intent secondintent = getIntent();
        String tedavi2 = secondintent.getStringExtra("tedavi");

        int ilkvirgul = tedavi2.indexOf(',');
        String veri_hastalik = tedavi2.substring(0, ilkvirgul);
        String veri_tedavi = tedavi2.substring(ilkvirgul + 1);
        // MainActivity'den gelen Python verisini ilk virgülden itibaren iki parçaya ayırmaya yarayan kod. Bu sayede hastalık ve tedavi ayrılabiliyor.


        hastalik = findViewById(R.id.tx_hastalik);
        hastalik.setText(veri_hastalik);
        tedavi = findViewById(R.id.tx_tedavi);
        tedavi.setText(veri_tedavi);
        // TextView'lere hastalığı ve tedaviyi yazdırmayı sağlayan kod bloğu.




    }

    public void OnButtonClicked(View view){
        Intent intent = new Intent(this, MainActivity.class);

        startActivity(intent);
        overridePendingTransition(R.anim.slide_in_left,R.anim.slide_out_right);
        // Butona bastığımızda normal animasyonumun tersi bir şekilde önceki sayfaya dönmeyi sağlayan fonksiyon.

    }


}