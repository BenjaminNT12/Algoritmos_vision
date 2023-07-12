library ieee;
use ieee.std_logic_1164.all;

entity sequence_detector is
  port (
    clk : in std_logic;
    reset : in std_logic;
    din : in std_logic;
    dout : out std_logic_vector(7 downto 0)
  );
end sequence_detector;

architecture behavioral of sequence_detector is
  signal state : std_logic_vector(7 downto 0) := (others => '0');

begin
  process (clk, reset)
  begin
    if reset = '1' then
      state <= (others => '0');
    elsif rising_edge(clk) then
      if din = '1' then
        state <= state + '1';
      else
        state <= state - '1';
      end if;
    end if;
  end process;

  dout <= state when state = "01101010" else (others => '0');
end behavioral;