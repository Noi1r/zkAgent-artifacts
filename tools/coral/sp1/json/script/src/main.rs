use lib::{Account, Transaction};
use sp1_sdk::{include_elf, utils, ProverClient, SP1ProofWithPublicValues, SP1Stdin};

const JSON_ELF: &[u8] = include_elf!("json-program");

fn main() {
    utils::setup_logger();

    let mut stdin = SP1Stdin::new();

    let data_str = r#"
            {
            "iss": "https://accounts.google.com",
            "azp": "5731200708710k7ga6ns79ie0jpg1ei6ip5vje2ostt6.apps.googleusercontent.com",
            "aud": "5731200708710k7ga6ns79ie0jpg1ei6ip5vje2ostt6.apps.googleusercontent.com",
            "sub": "115379862439230851421",
            "nonce": "UZ2BZTyy4vyLnXoMc60cuv8AXGI",
            "nbf": 1748983762,
            "iat": 1748984062,
            "exp": 1748987662,
            "jti": "0bcd861dc12b697475723359de2d9ce39f38ae98"
            }"#
    .to_string();
    let key = "net_worth".to_string();

    let initial_account_state = Account { account_name: "John".to_string(), balance: 200 };
    let transactions = vec![
        Transaction { from: "John".to_string(), to: "Uma".to_string(), amount: 50 },
        Transaction { from: "Uma".to_string(), to: "John".to_string(), amount: 100 },
    ];

    stdin.write(&data_str);

    let client = ProverClient::from_env();
    let (pk, vk) = client.setup(JSON_ELF);
    let mut proof = client.prove(&pk, &stdin).run().expect("proving failed");

    let val = proof.public_values.read::<String>();
    println!("Return should be 1 and is {}", val);

    client.verify(&proof, &vk).expect("verification failed");

    proof.save("proof-with-io.bin").expect("saving proof failed");
    let deserialized_proof =
        SP1ProofWithPublicValues::load("proof-with-io.bin").expect("loading proof failed");

    client.verify(&deserialized_proof, &vk).expect("verification failed");

    println!("successfully generated and verified proof for the program!")
}
