package utils;

import weka.core.Instance;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.text.Normalizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MyStringToWordVector extends StringToWordVector {

    @Override
    public boolean input(Instance instance) throws Exception {

        String text = instance.stringValue(0);
        text = normalizeText(text);
        instance.setValue(0,text);

        return super.input(instance);
    }

    private String normalizeText(String text){
        Pattern charsPunctuationPattern = Pattern.compile("[\\d:,\"\'\\`\\_\\|?!\n\r@;]+");
        String input_text = charsPunctuationPattern.matcher(text.trim().toLowerCase()).replaceAll("");
        input_text = input_text.replaceAll("\\{.*?\\}", "");
        input_text = input_text.replaceAll("\\[.*?\\]", "");
        input_text = input_text.replaceAll("\\(.*?\\)", "");
        input_text = input_text.replaceAll("[^A-Za-z0-9(),!?@\'\\`\"\\_\n]", " ");
        input_text = input_text.replaceAll("[/]"," ");
        input_text = input_text.replaceAll("\\d", "");

        return input_text;
    }
}