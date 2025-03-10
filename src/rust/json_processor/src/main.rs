use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{Seek, self, BufReader, Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::env;

#[derive(Serialize, Deserialize, Debug)]
struct Definition {
    name: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Sample {
    #[serde(rename = "genTime")]
    gen_time: i64,
    #[serde(rename = "vFlt")]
    v_flt: f64,
    #[serde(rename = "rawVFlt")]
    r_flt: f64,
    #[serde(rename = "parentId")]
    parent_id: i64,
    #[serde(rename = "validity")]
    validity: String
}

#[derive(Serialize, Deserialize, Debug)]
struct TopLevel {
    #[serde(rename = "definition")]
    definition: Definition,
    samples: Vec<Sample>,
}

fn process_json_file<P: AsRef<Path>>(
    input_path: P,
    output_base: P,
    max_lines_per_file: usize,
) -> Result<(), Box<dyn Error>> {
    let file = File::open(input_path.as_ref())?;
    let reader = BufReader::new(file);

    let mut first_char_buffer = [0; 1];
    let mut buffered_reader = BufReader::new(reader);
    buffered_reader.read_exact(&mut first_char_buffer)?;
    let mut reader = buffered_reader.into_inner();
    reader.seek(io::SeekFrom::Start(0))?;
    let first_char = first_char_buffer[0] as char;

    if first_char != '{' {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid JSON: must start with '{'",
        )));
    }

    let top_level: TopLevel = serde_json::from_reader(reader)?;
    let column_name = &top_level.definition.name;

    let mut file_counter = 1;
    let mut line_counter = 0;
    let mut output_file = create_output_file(&output_base, file_counter)?;
    writeln!(output_file, "Timestamp,{}", column_name)?;

    for sample in top_level.samples {
        
        if sample.validity != "VALID" {
            continue;
        }

        if line_counter >= max_lines_per_file {
            output_file = create_output_file(&output_base, file_counter + 1)?;
            file_counter += 1;
            line_counter = 0;
            writeln!(output_file, "Timestamp,{}", column_name)?;
        }

        writeln!(
            output_file,
            "{},{}",
            sample.gen_time, sample.v_flt
        )?;
        line_counter += 1;
    }

    Ok(())
}

fn create_output_file<P: AsRef<Path>>(
    output_base: P,
    file_counter: usize,
) -> Result<File, std::io::Error> {
    let output_path = output_base.as_ref().with_extension(format!("{}.csv", file_counter));
    File::create(output_path)
}

fn main() {
     let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <input_json_file> <output_csv_base_name> <max_lines_per_file>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_filename = &args[1];
    let output_base_name = &args[2];
    let max_lines_per_file = match usize::from_str(&args[3]) {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: Invalid number of lines per file: {}", args[3]);
            std::process::exit(1);
        }
    };


    if let Err(e) = process_json_file(input_filename, output_base_name, max_lines_per_file) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}