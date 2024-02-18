package com.berkzerey.aidoc;

import static android.content.Context.MODE_PRIVATE;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDialogFragment;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.os.Bundle;

public class PopUpActivity extends AppCompatDialogFragment {

    public static final String HELP_POPUP = "help_popup";
    public static final String MAX_POPUP = "max_popup";
    public static final String MIN_POPUP = "min_popup";
    public static final String FIRST_POPUP = "first_popup";
    public static final String LOADING_POPUP = "loading_popup";
    // Burada oluşturacağım PopUp'ları tanımladım.

    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        String popupType = getArguments().getString("popupType");

        switch (popupType) {
            case HELP_POPUP:
                builder.setTitle("YARDIM")
                        .setMessage("Sahip olduğunuz semptomları seçtikten sonra aşağıya kaydırıp 'TARA' " +
                                "butonuna basın. Size ihtimal dahilinde hangi hastalığa sahip olduğunuzu belirtecektir.")
                        .setPositiveButton("Tamam", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                            }
                        });
                break;
            case MAX_POPUP:
                builder.setTitle("Uyarı")
                        .setMessage("Seçilebilecek maksimum semptom sayısına ulaşıldı.")
                        .setPositiveButton("Tamam", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                            }
                        });
                break;

            case MIN_POPUP:
                builder.setTitle("Uyarı")
                        .setMessage("Lütfen en az 2 adet semptom seçiniz.")
                        .setPositiveButton("Tamam", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {

                            }
                        });
                break;
            case FIRST_POPUP:
                builder.setTitle("Sorumluluk Beyanı")
                        .setMessage("Verdiğim bilgilerin doğru olduğunu ve ileride karşıma çıkabilecek bütün sorumlulukları kabul ediyorum.")
                        .setPositiveButton("Evet", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                SharedPreferences prefs = getActivity().getSharedPreferences("prefs",MODE_PRIVATE);
                                SharedPreferences.Editor editor = prefs.edit();
                                editor.putBoolean("firstStart",false);
                                editor.apply();
                                // Burada SharedPreferences kullanarak sorumluluk beyanı PopUp'ı yolluyorum. SharedPreferences sayesinde
                                // kullanıcının uygulamaya ilk defa girip girmediğini anlayabiliyorum.
                            }
                        }
                        ).setNegativeButton("Hayır", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                System.exit(0);
                            }
                        });
                break;

            case LOADING_POPUP:

                builder.setMessage("HESAPLANIYOR");

        }

        return builder.create();
    }

}