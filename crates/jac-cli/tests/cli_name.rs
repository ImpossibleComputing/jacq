use std::path::Path;
use std::process::Command;

fn jacq(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jacq"))
        .args(args)
        .output()
        .expect("run jacq")
}

#[test]
fn executable_and_user_facing_name_are_jacq() {
    assert_eq!(
        Path::new(env!("CARGO_BIN_EXE_jacq"))
            .file_stem()
            .and_then(|name| name.to_str()),
        Some("jacq")
    );

    let help = jacq(&["--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.starts_with("jacq —"), "{help}");
    assert!(help.contains("Usage: jacq [OPTIONS] [FILE]"), "{help}");

    let version = jacq(&["--version"]);
    assert!(version.status.success());
    let version = String::from_utf8(version.stdout).unwrap();
    assert!(version.starts_with("jacq "), "{version}");

    let missing_arg = jacq(&["--validate"]);
    assert!(!missing_arg.status.success());
    assert_eq!(
        String::from_utf8(missing_arg.stderr).unwrap(),
        "jacq --validate: expected a .jac file\n"
    );

    let missing_file = jacq(&["--validate", "this-file-does-not-exist.jac"]);
    assert!(!missing_file.status.success());
    assert!(String::from_utf8(missing_file.stderr)
        .unwrap()
        .starts_with("jacq: cannot read this-file-does-not-exist.jac:"));
}
