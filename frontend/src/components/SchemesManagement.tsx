import { FC, useEffect, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { Modal } from 'react-bootstrap';
import { IoDuplicate } from 'react-icons/io5';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { useAddScheme, useDeleteScheme, useDuplicateScheme, useRenameScheme } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { getRandomName } from '../core/utils';
import { SchemeModel } from '../types';
/*
 * Select current scheme
 */

interface SchemeManagementProps {
  projectSlug: string;
  canEdit?: boolean;
  username?: string | null;
}

export const SelectCurrentScheme: FC<{ username?: string | null }> = ({ username }) => {
  const { notify } = useNotifications();

  // get element from the context
  const {
    appContext: { currentProject, currentScheme },
    setAppContext,
  } = useAppContext();

  const availableSchemes = useMemo(() => {
    return currentProject ? Object.keys(currentProject.schemes.available) : [];
  }, [currentProject]);

  // manage scheme selection
  useEffect(() => {
    // case of there is no selected scheme and schemes are available
    const nonDefaultSchemes = availableSchemes.filter((element) => element !== 'default');
    if (!currentScheme && nonDefaultSchemes.length > 0) {
      // if there is a previous scheme in the user history select it
      if (
        username &&
        currentProject?.users.last_schemes[username] &&
        nonDefaultSchemes.includes(currentProject.users.last_schemes[username])
      )
        setAppContext((state) => ({
          ...state,
          currentScheme: currentProject.users.last_schemes[username],
        }));
      else
        setAppContext((state) => ({
          ...state,
          currentScheme: nonDefaultSchemes[0],
        }));
      notify({
        type: 'success',
        message: `Scheme selected`,
      });
    }
    // case of the scheme have been deleted
    if (availableSchemes[0] && currentScheme && !availableSchemes.includes(currentScheme)) {
      setAppContext((state) => ({
        ...state,
        currentScheme: availableSchemes[0],
      }));
    }
  }, [currentScheme, availableSchemes, setAppContext, notify, currentProject, username]);

  // put the current scheme in the context on change
  const handleSelectScheme = (selectedOption: { value: string; label: string } | null) => {
    setAppContext((state) => ({
      ...state,
      currentScheme: selectedOption ? selectedOption.value : availableSchemes[0],
    }));
    notify({
      type: 'success',
      message: `Scheme ${selectedOption ? selectedOption.value : availableSchemes[0]} selected`,
    });
  };

  return (
    <div className="row">
      <div className="input-group mb-3">
        <span className="input-group-text d-none d-md-inline bg-primary" style={{ color: 'white' }}>
          Current Scheme
        </span>
        <Select
          id="scheme-selected"
          className="flex-grow-1"
          options={availableSchemes.map((element) => ({
            value: element,
            label: element,
          }))}
          value={{
            value: currentScheme || availableSchemes[0],
            label: currentScheme || availableSchemes[0],
          }}
          onChange={handleSelectScheme}
        />
      </div>
    </div>
  );
};

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<SchemeManagementProps> = ({
  projectSlug,
  canEdit,
  username,
}) => {
  // get element from the context
  const {
    appContext: { currentProject, currentScheme, reFetchCurrentProject },
    setAppContext,
  } = useAppContext();

  const availableSchemes = currentProject ? Object.keys(currentProject.schemes.available) : [];

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({
    defaultValues: {
      name: getRandomName('Scheme'),
    },
  });

  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, currentScheme || null);
  const renameScheme = useRenameScheme(projectSlug, currentScheme || '');
  const duplicateScheme = useDuplicateScheme(projectSlug, currentScheme || '');

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const [showRename, setShowRename] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [newSchemeName, setNewSchemeName] = useState('New name');

  // action to create the new scheme
  const createNewScheme: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      await addScheme(formData.name, formData.kind || 'multiclass');
      if (reFetchCurrentProject) reFetchCurrentProject();
      notify({ type: 'success', message: `Scheme ${formData.name} created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    setShowCreateNewScheme(false);
  };

  // action to delete scheme
  const deleteSelectedScheme = async () => {
    //TODO: try catch and throw
    await deleteScheme();

    // set another scheme
    const otherSchemes = availableSchemes.filter((element) => element !== currentScheme);
    setAppContext((state) => ({
      ...state,
      currentScheme: otherSchemes[0],
    }));
    notify({ type: 'success', message: 'Scheme changed to ' + otherSchemes[0] });

    if (reFetchCurrentProject) reFetchCurrentProject();
  };

  return (
    <div>
      <div className="mt-3 col-12 d-flex">
        <SelectCurrentScheme username={username} />
        {canEdit && (
          <div className="mx-2">
            <button
              onClick={() => setShowCreateNewScheme(!showCreateNewScheme)}
              className="btn btn-sm p-1 addscheme"
            >
              <FaPlusCircle size={20} />
              <Tooltip anchorSelect=".addscheme" place="top">
                Add empty scheme
              </Tooltip>
            </button>
            <button
              onClick={() => setShowRename(!showRename)}
              className="btn btn-sm p-1 renamescheme"
            >
              <MdDriveFileRenameOutline size={20} />
              <Tooltip anchorSelect=".renamescheme" place="top">
                Rename current scheme
              </Tooltip>
            </button>
            <button onClick={() => duplicateScheme()} className="btn btn-sm p-1 duplicatescheme">
              <IoDuplicate size={20} />
              <Tooltip anchorSelect=".duplicatescheme" place="top">
                Duplicate current scheme
              </Tooltip>
            </button>
            <button
              onClick={() => setShowDelete(!showDelete)}
              className="btn btn-sm p-1 deletescheme"
            >
              <FaRegTrashAlt size={20} />
              <Tooltip anchorSelect=".deletescheme" place="top">
                Delete current scheme
              </Tooltip>
            </button>
          </div>
        )}
      </div>
      <Modal
        show={showCreateNewScheme}
        id="addschememodal"
        onHide={() => setShowCreateNewScheme(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Add a new scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmit(createNewScheme)}>
            <input
              className="form-control"
              id="scheme_name"
              type="text"
              {...register('name')}
              placeholder="Enter new scheme name"
            />
            <label htmlFor="scheme-selected" style={{ whiteSpace: 'nowrap', marginRight: '10px' }}>
              Type
            </label>
            <select id="scheme_kind" className="form-select" {...register('kind')}>
              <option value="multiclass">Multiclass</option>
              <option value="multilabel">
                Multilabel (experimental - only some features implemented)
              </option>
              <option value="span">Span (experimental - only annotation)</option>
            </select>
            <button className="btn btn-primary btn-validation">Create</button>
          </form>
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="renamescheme" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename the current scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <input
            className="form-control my-2"
            id="scheme_rename"
            type="text"
            placeholder="Enter new scheme name"
            value={newSchemeName}
            onChange={(e) => setNewSchemeName(e.target.value)}
          />
          <button
            className="btn btn-primary"
            onClick={() => {
              renameScheme(newSchemeName);
              setShowRename(false);
            }}
          >
            Rename
          </button>
        </Modal.Body>
      </Modal>
      <Modal show={showDelete} id="deletescheme" onHide={() => setShowDelete(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Delete the current scheme</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            Are you sure you want to delete the scheme <b>{currentScheme}</b>?
          </p>
          <button
            className="btn btn-danger"
            onClick={() => {
              deleteSelectedScheme();
              setShowDelete(false);
            }}
          >
            Delete
          </button>
        </Modal.Body>
      </Modal>
    </div>
  );
};
