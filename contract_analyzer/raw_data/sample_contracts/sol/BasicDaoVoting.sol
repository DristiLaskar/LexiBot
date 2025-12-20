pragma solidity ^0.8.0;

contract SimpleDAO {
    address public chairperson;
    mapping(address => bool) public members;
    mapping(bytes32 => uint) public proposals;

    constructor() {
        chairperson = msg.sender;
    }

    function addMember(address member) external {
        require(msg.sender == chairperson, "Only chairperson");
        members[member] = true;
    }

    function vote(bytes32 proposal) external {
        require(members[msg.sender], "Not a member");
        proposals[proposal] += 1;
    }
}