package utils;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;

public class MergeFiles {

    public static void main(String[] args) throws Exception {
        // Input files
        List<Path> inputs = Arrays.asList(
                Paths.get("./dataset/asset"),
                Paths.get("./dataset/control"),
                Paths.get("./dataset/implicit"),
                Paths.get("./dataset/threat")
        );

        // Output file
        Path output = Paths.get("./dataset/security_keywords.txt");

        // Charset for read and write
        Charset charset = StandardCharsets.UTF_8;

        // Join files (lines)
        for (Path path : inputs) {
            List<String> lines = Files.readAllLines(path, charset);
            Files.write(output, lines, charset, StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        }
    }
}
