package com.berkzerey.aidoc;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ScrollView;
import android.widget.SearchView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    TextView tv, checkboxCounter;
    Button bt_tara,bt_asagi;
    private int selectedCount = 0;
    private int maximumCount = 17;

    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.actionbar_menu, menu); //Üst barımızda üç noktaya basınca menü açmayı sağlar.
        return true;
    }

    public boolean onOptionsItemSelected(@NonNull MenuItem item) { // Menüdeki elemanları seçme olayımız

        switch (item.getItemId()){
            case R.id.menu_help:
                showPopup(PopUpActivity.HELP_POPUP); // Yardım PopUp'ımızı açıyor.
                break;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getSupportActionBar().setDisplayShowTitleEnabled(false); //Uygulamanın adının ActionBar'da gözükmesini istemiyorum.

        bt_tara = findViewById(R.id.bas);
        bt_asagi = findViewById(R.id.bt_asagikaydir);
        checkboxCounter =  findViewById(R.id.semptom_sayaci);
        SearchView searchView = findViewById(R.id.sv_semptomara);

        SharedPreferences prefs = getSharedPreferences("prefs",MODE_PRIVATE);
        boolean firstStart = prefs.getBoolean("firstStart",true);
        if(firstStart){showPopup(PopUpActivity.FIRST_POPUP);}
        // Uygulama ilk kez başlatıldığında sorumluluk beyanı bildirimi veren kod bloğu, eğer kabul edilmezse uygulama kapanıyor.

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                // Üstteki SearchBar'da text yazıldıktan sonra entere bastığımızda ne olacağını gösteriyor, ben herhangi bir şey yapmayacağım.
                return false;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                // SearchBar'daki text her değiştiğinde Checkbox'ları o text'e göre filtreliyor.
                filterSymptoms(newText);
                return true;
            }
        });

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this)); //Python'u başlamadıysa başlatan kod bloğu.
        }

        final Python py = Python.getInstance();

        CompoundButton.OnCheckedChangeListener checkBoxListener = new CompoundButton.OnCheckedChangeListener() {
            @SuppressLint("ResourceAsColor")
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked && selectedCount == maximumCount) {
                    showPopup(PopUpActivity.MAX_POPUP); //Maksimuma ulaştınız uyarısını veren PopUp'ı gösteriyor.

                    buttonView.setChecked(false); // Eğer maksimum sayıya ulaşıldıysa son seçileni geri alıyor.
                }
                else {
                    selectedCount = isChecked ? selectedCount + 1 : selectedCount - 1;
                    if (selectedCount < 0) {
                        selectedCount = 0;
                    }

                    checkboxCounter.setText(selectedCount + "/" + maximumCount); // Semptom saymaya yarayan metnimizi güncelliyor.

                    if (selectedCount >= maximumCount) {
                        checkboxCounter.setTextColor(ContextCompat.getColor(MainActivity.this, R.color.aidoc_red));
                        //Eğer semptom sayısı maksimuma ulaştıysa renk kırmızı oluyor.
                    } else {
                        checkboxCounter.setTextColor(ContextCompat.getColor(MainActivity.this, R.color.aidoc_turquoise));
                        //Maksimum olmadığı her sayıda metin turkauaz renk oluyor.
                    }
                }
            }
        };


        List<Integer> symptomIds = SymptomsActivity.getSymptoms();

        for (int checkboxId : symptomIds) {
            CheckBox checkbox = findViewById(checkboxId);
            checkbox.setOnCheckedChangeListener(checkBoxListener);
        }

        bt_tara.setOnClickListener(view -> {

            bt_tara.setEnabled(false);

            List<Integer> kullanicisemptomlari = SymptomsActivity.getSymptoms(); //Semptomları al

            int selectedSymptomCount = 0;

            for (int checkboxId : kullanicisemptomlari) {
                CheckBox checkbox = findViewById(checkboxId);
                if (checkbox.isChecked()) {
                    selectedSymptomCount++; //Seçili semptom başına arttır.
                }
                checkbox.setEnabled(false);
            }

            if (selectedSymptomCount < 2) {
                // En az 2 semptom seçilmedi, kullanıcıyı uyar
                showPopup(PopUpActivity.MIN_POPUP);
                bt_tara.setEnabled(true);
                for (int checkboxId : kullanicisemptomlari) {
                    CheckBox checkbox = findViewById(checkboxId);
                    checkbox.setEnabled(true);
                }
            }
            else {

                int[] symptomArray = new int[kullanicisemptomlari.size()];
                for (int i = 0; i < kullanicisemptomlari.size(); i++) {
                    // Her bir checkbox'ın durumunu alarak 1 veya 0 değerini belirle
                    CheckBox checkbox = findViewById(kullanicisemptomlari.get(i));
                    symptomArray[i] = checkbox.isChecked() ? 1 : 0;
                }

                showPopup(PopUpActivity.LOADING_POPUP); // İşlem yapılırken yüklenioyr PopUp'ı gösteriyor

                new Handler().postDelayed(() -> {
                    PyObject pyo = py.getModule("ann_pragnoise"); // ann_pragnoise.py isimli dosyayla bağlantı kuran kod
                    PyObject obj = pyo.callAttr("main", symptomArray); // main fonksiyonuna değişken gönderip fonksiyonu çalıştırıyor
                    String hastaliktedavi = obj.toString(); // main dosyasından return edilen değeri atıyor

                    // İşlem tamamlandıktan sonra checkbox'ları temizle
                    for (int checkboxId : kullanicisemptomlari) {
                        CheckBox checkbox = findViewById(checkboxId);
                        checkbox.setChecked(false);
                    }

                    Intent intent = new Intent(this, NextActivity.class);
                    intent.putExtra("tedavi", hastaliktedavi);
                    startActivity(intent);
                    // NextActivity sayfama değer yolluyorum

                    bt_tara.setEnabled(true);
                    for (int checkboxId : kullanicisemptomlari) {
                        CheckBox checkbox = findViewById(checkboxId);
                        checkbox.setEnabled(true);
                    }

                }, 2000); // İşlem yapılırken 2 saniye gecikmeli yapıyorum ki geçiş daha akıcı olsun.


            }
        });


        bt_asagi.setOnClickListener(view -> { // Bu butona bastığında uygulamayı en aşağıya kaydırıyor.
            ScrollView scrollView = findViewById(R.id.scv_scroll);
            scrollView.fullScroll(View.FOCUS_DOWN);
        });

    }

    private void showPopup(String popupType) { // PopUp göstermemizi sağlayan fonksiyon.
        Bundle bundle = new Bundle();
        bundle.putString("popupType", popupType);

        PopUpActivity popUpActivity = new PopUpActivity();
        popUpActivity.setArguments(bundle);
        popUpActivity.show(getSupportFragmentManager(), popupType);
    }

    private void filterSymptoms(String query) { // SearchBar'da arattığımızda semptomları filtrelemeye yarayan fonksiyon.
        List<Integer> symptomIds = SymptomsActivity.getSymptoms();

        for (int checkboxId : symptomIds) {
            CheckBox checkbox = findViewById(checkboxId);
            String symptomText = checkbox.getText().toString().toLowerCase();

            if (symptomText.contains(query.toLowerCase())) {
                checkbox.setVisibility(View.VISIBLE);
            } else {
                checkbox.setVisibility(View.GONE);
            }
        }
    }

}