import { FC, useEffect, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { IoDuplicate } from 'react-icons/io5';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import { useAddScheme, useDeleteScheme, useDuplicateScheme, useRenameScheme } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { SchemeModel } from '../types';

/*
 * Select current scheme
 */

interface SchemeManagementProps {
  projectSlug: string;
  deactivateModifications?: boolean;
}

export const SelectCurrentScheme: FC = () => {
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
      setAppContext((state) => ({
        ...state,
        currentScheme: nonDefaultSchemes[0],
      }));
      notify({
        type: 'success',
        message: `Scheme ${nonDefaultSchemes[0]} selected`,
      });
    }
    // case of the scheme have been deleted
    if (availableSchemes[0] && currentScheme && !availableSchemes.includes(currentScheme)) {
      setAppContext((state) => ({
        ...state,
        currentScheme: availableSchemes[0],
      }));
    }
  }, [currentScheme, availableSchemes, setAppContext, notify]);

  // put the current scheme in the context on change
  const handleSelectScheme = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setAppContext((state) => ({
      ...state,
      currentScheme: event.target.value,
    }));
    notify({ type: 'success', message: 'Scheme selected' });
  };

  console.log('currentScheme', currentScheme);

  return (
    <div className="row">
      <div className="d-flex align-items-center mb-3">
        <label htmlFor="scheme-selected" style={{ whiteSpace: 'nowrap', marginRight: '10px' }}>
          Current
        </label>
        <select
          id="scheme-selected"
          className="form-select"
          onChange={handleSelectScheme}
          value={currentScheme ? currentScheme : availableSchemes[0]}
        >
          {availableSchemes.map((element) => (
            <option key={element} value={element}>
              {element}
            </option>
          ))}{' '}
        </select>
      </div>
    </div>
  );
};

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<SchemeManagementProps> = ({
  projectSlug,
  deactivateModifications,
}) => {
  // get element from the context
  const {
    appContext: { currentProject, currentScheme, reFetchCurrentProject },
    setAppContext,
  } = useAppContext();

  const availableSchemes = currentProject ? Object.keys(currentProject.schemes.available) : [];

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, currentScheme || null);
  const renameScheme = useRenameScheme(projectSlug, currentScheme || '');
  const duplicateScheme = useDuplicateScheme(projectSlug, currentScheme || '');

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const [showRename, setShowRename] = useState(false);
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
    setShowCreateNewScheme(!showCreateNewScheme);
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
    <div className="container-fluid m-3">
      <div className="d-flex flex-wrap align-items-center">
        <div className="mt-3 col-12 col-md-4">
          <SelectCurrentScheme />
        </div>
        {!deactivateModifications && (
          <>
            <button onClick={deleteSelectedScheme} className="btn btn-primary mx-2">
              <FaRegTrashAlt size={20} /> <span className="d-none d-sm-inline ms-2">Delete</span>
            </button>
            <button
              onClick={() => setShowCreateNewScheme(!showCreateNewScheme)}
              className="btn btn-primary"
            >
              <FaPlusCircle size={20} /> <span className="d-none d-sm-inline ms-2">Add</span>
            </button>
            <button onClick={() => setShowRename(!showRename)} className="btn btn-primary mx-2">
              <MdDriveFileRenameOutline size={20} />{' '}
              <span className="d-none d-sm-inline ms-2">Rename</span>
            </button>
            <button onClick={() => duplicateScheme()} className="btn btn-primary">
              <IoDuplicate size={20} /> <span className="d-none d-sm-inline ms-2">Duplicate</span>
            </button>
          </>
        )}
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewScheme && (
            <div className="row">
              <div className="col-12 col-md-6 m-2 alert alert-primary">
                <form onSubmit={handleSubmit(createNewScheme)}>
                  <input
                    className="form-control"
                    id="scheme_name"
                    type="text"
                    {...register('name')}
                    placeholder="Enter new scheme name"
                  />
                  <label
                    htmlFor="scheme-selected"
                    style={{ whiteSpace: 'nowrap', marginRight: '10px' }}
                  >
                    Type
                  </label>
                  <select id="scheme_kind" className="form-select" {...register('kind')}>
                    <option value="multiclass">Multiclass</option>
                    <option value="multilabel">Multilabel (expertimental)</option>
                  </select>
                  <button className="btn btn-primary btn-validation">Create</button>
                </form>
              </div>
            </div>
          )
        }
        {
          // only display if click on the rename button
          showRename && (
            <div className="row">
              <div className="col-8 col-md-4 d-flex m-2 alert alert-primary">
                <input
                  className="form-control me-2"
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
              </div>
            </div>
          )
        }
      </div>
    </div>
  );
};
