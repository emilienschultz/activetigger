# Documentation

This documentation details the different logics and functionalities of Active Tigger. It is a work in progress. It will be updated as the project evolves.


## General comments

Only one process allowed in the same time by user. There is two types of processes in Active Tigger:

- CPU processes: they are used to prepare data, train quick models, etc.
- GPU processes: these processes are run on the GPU. The server can only run a limited number of GPU processes at the same time.

In both case, if the number of processes is too high, they will be queued.

## Accounts

A user can have the following status :

- **root**: can create and manage projects, users, and all data. Can access all projects and the /monitor page.
- **manager**: can create and manage projects where he.she has complete rights. Can add users to the projects.
- **annotator**: can only annotate on projects where he.she have been added. The frontend is simplified for this role.

User have also a relational status for a specific project:

- **manager**: can manage the project, add users, and access all data.
- **annotator**: can only annotate on the project without changing the project settings (schemes, labels, models, etc.).

For the moment, there is no management at the scheme level.

## Create a project

## Prepare labels and features

## Explore the data

## Annotate

## Fine-tune a BERT model

## Test model

The test set:
- Created on the beginning of the project 
- Uploaded latter

Once activated, the test mode :
- Deactivate for the user the choice of scheme, label management
- Allow only annotation for the test set
- Allow to explore the test set


## Export data / models

