module PE (
    input clk,
    input reset,

    input load_weight, //signal to tell whether to load the weight.
    input signed [7:0] weight_in, //signed 8 bit integer

    input signed [7:0] data_in, //signed 8 bit integer
    input signed [17:0] psum_in, //signed 18 bit integer

    output reg signed [7:0] data_out, //signed 8 bit integer
    output reg signed [17:0] psum_out //signed 18 bit integer
);

    reg signed [7:0] weight;

    wire signed [15:0] product;
    assign product = weight * data_in;

    always @(posedge clk) begin
        if (reset) begin
            weight   <= 0;
            data_out <= 0;
            psum_out <= 0;
        end
        else begin
            // Load weight when requested
            if (load_weight)
                weight <= weight_in;

            // Pass input to neighboring PE
            data_out <= data_in;

            // Multiply and accumulate
            psum_out <= psum_in +
                        {{2{product[15]}}, product};
        end
    end
endmodule