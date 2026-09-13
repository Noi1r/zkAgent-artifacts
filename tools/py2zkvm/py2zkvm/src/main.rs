mod transpiler;
mod parser;
mod generator;
mod project;

use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "py2zkvm")]
#[command(about = "Python to SP1 zkVM transpiler - converts Python scripts to provable SP1 programs")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {

    Transpile {

        #[arg(short, long)]
        input: PathBuf,


        #[arg(short, long, default_value = "sp1_output")]
        output: PathBuf,


        #[arg(short, long)]
        main_fn: Option<String>,


        #[arg(long)]
        precompiles: bool,
    },


    Convert {

        #[arg(short, long)]
        input: PathBuf,


        #[arg(short, long)]
        output: Option<PathBuf>,
    },


    Analyze {

        #[arg(short, long)]
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Transpile { input, output, main_fn, precompiles } => {
            println!("=== py2zkvm: Python to SP1 zkVM Transpiler ===\n");

            println!("[1/4] Transpiling Python to Rust...");
            let rust_code = transpiler::transpile_python(&input)?;
            println!("      Generated {} bytes of Rust code", rust_code.len());

            println!("[2/4] Analyzing Rust functions...");
            let functions = parser::parse_rust_functions(&rust_code)?;
            println!("      Found {} functions:", functions.len());
            for f in &functions {
                println!("        - {}({}) -> {}", f.name,
                    f.params.iter().map(|(n, t)| format!("{}: {}", n, t)).collect::<Vec<_>>().join(", "),
                    f.return_type);
            }

            let main_function = if let Some(ref name) = main_fn {
                functions.iter().find(|f| &f.name == name)
                    .ok_or_else(|| anyhow::anyhow!("Function '{}' not found", name))?
            } else {
                functions.first()
                    .ok_or_else(|| anyhow::anyhow!("No functions found in Python file"))?
            };
            if main_function.params.iter().any(|(_, t)| parser::contains_generic_param(t))
                || parser::contains_generic_param(&main_function.return_type) {
                anyhow::bail!(
                    "Function '{}' uses generic type parameters (e.g. T). Add explicit Python type annotations so depyler produces concrete types.",
                    main_function.name
                );
            }
            println!("\n[3/4] Generating SP1 project for function '{}'...", main_function.name);

            let config = generator::Sp1Config {
                main_function: main_function.clone(),
                all_functions: functions,
                rust_code: rust_code.clone(),
                precompiles,
            };
            project::generate_sp1_project(&output, &config)?;

            println!("[4/4] SP1 project generated at: {}", output.display());
            println!("\n=== Build Instructions ===");
            println!("cd {}/script", output.display());
            println!("cargo run --release");

            Ok(())
        }

        Commands::Convert { input, output } => {
            let rust_code = transpiler::transpile_python(&input)?;

            if let Some(output_path) = output {
                std::fs::write(&output_path, &rust_code)?;
                println!("Rust code written to: {}", output_path.display());
            } else {
                println!("{}", rust_code);
            }

            Ok(())
        }

        Commands::Analyze { input } => {
            println!("Analyzing: {}\n", input.display());

            let rust_code = transpiler::transpile_python(&input)?;
            let functions = parser::parse_rust_functions(&rust_code)?;

            println!("Found {} provable functions:\n", functions.len());
            for f in &functions {
                println!("Function: {}", f.name);
                println!("  Parameters:");
                for (name, typ) in &f.params {
                    let zkvm_compat = parser::check_zkvm_compatibility(typ);
                    let status = if zkvm_compat { "✓" } else { "⚠" };
                    println!("    {} {}: {}", status, name, typ);
                }
                println!("  Returns: {}", f.return_type);
                println!("  zkVM Compatible: {}\n",
                    if f.params.iter().all(|(_, t)| parser::check_zkvm_compatibility(t)) { "Yes" } else { "Partial" });
            }

            Ok(())
        }
    }
}
