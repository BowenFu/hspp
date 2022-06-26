#!/bin/sh

cd include/hspp
awk 1 range.h data.h typeclass.h do_notation.h parser.h > ../hspp_single_header.h
