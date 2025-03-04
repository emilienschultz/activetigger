import { FC, useEffect, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { useAddScheme, useDeleteScheme } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { SchemeModel } from '../types';

/*
 * Select current scheme
 */

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
    if (!currentScheme && availableSchemes.length > 0) {
      setAppContext((state) => ({
        ...state,
        currentScheme: availableSchemes[0],
      }));
      notify({ type: 'success', message: `Scheme ${currentScheme} selected by default` });
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

  return (
    <div className="row">
      <div className="d-flex align-items-center mb-3">
        <label htmlFor="scheme-selected" style={{ whiteSpace: 'nowrap', marginRight: '10px' }}>
          Scheme to use
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

export const SchemesManagement: FC<{ projectSlug: string }> = ({ projectSlug }) => {
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

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const handleIconClick = () => {
    setShowCreateNewScheme(!showCreateNewScheme);
  };

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
        <div className="mt-3">
          <SelectCurrentScheme />
        </div>
        <button onClick={deleteSelectedScheme} className="btn btn-primary mx-2">
          <FaRegTrashAlt size={20} /> Delete
        </button>
        <button onClick={handleIconClick} className="btn btn-primary">
          <FaPlusCircle size={20} /> Add
        </button>
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewScheme && (
            <div className="row">
              <form onSubmit={handleSubmit(createNewScheme)}>
                <div className="col-4">
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
                </div>
              </form>
            </div>
          )
        }
      </div>
    </div>
  );
};
