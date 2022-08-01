# rust-vivaldi ![status](https://github.com/euforia/rust-vivaldi/actions/workflows/rust.yml/badge.svg)

rust-vivaldi is an implementation of the vivaldi client in rust.  It is based on the Go implementation of the vivaldi client at [hexablock/vivaldi](http://github.com/hexablock/vivaldi) which in turn is based on the client in [hashicorp/serf](http://github.com/hashicorp/serf).

## Deviations

- This implemententation is currently not thread-safe.
- Certain unit tests are disabled due to computational differences in rounding.