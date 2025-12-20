// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleOracle {
    address public owner;
    uint public price;

    constructor() {
        owner = msg.sender;
    }

    function updatePrice(uint _price) external {
        require(msg.sender == owner, "Only owner");
        price = _price;
    }

    function getPrice() external view returns (uint) {
        return price;
    }
}
