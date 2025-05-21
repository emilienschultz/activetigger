import { FC, useState } from 'react';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { FaEdit } from 'react-icons/fa';
import { useAddLabel, useDeleteLabel, useRenameLabel } from '../core/api';
import { useNotifications } from '../core/notifications';

import { FaCheck } from 'react-icons/fa';

/**
 * Component to manage one label
 */

interface LabelsManagementProps {
  projectSlug: string | null;
  currentScheme: string | null;
  availableLabels: string[];
  kindScheme: string;
  reFetchCurrentProject: () => void;
}

export const LabelCard: FC<{
  label: string;
  removeLabel: (label: string) => void;
  renameLabel: (formerLabel: string, newLabel: string) => void;
}> = ({ label, removeLabel, renameLabel }) => {
  const [showRename, setShowRename] = useState(false);
  const [newLabel, setNewLabel] = useState(label);
  return (
    <tr key={label} className="border-b hover:bg-gray-50">
      <td className="px-4 py-3">{label}</td>
      <td className="px-4 py-3 text-center">0</td>
      <td className="flex justify-center gap-4">
        <div
          title="Delete"
          onClick={() => removeLabel(label)}
          className="cursor-pointer trash-wrapper"
        >
          <FaRegTrashAlt />
        </div>
      </td>
      <td className="flex justify-center gap-4">
        <div title="Rename" onClick={() => setShowRename(!showRename)} className="cursor-pointer">
          <FaEdit />
        </div>
      </td>
      {showRename && (
        <td>
          <div className="d-flex align-items-center">
            <input
              type="text"
              className="form-control"
              placeholder="Enter new label"
              onChange={(e) => setNewLabel(e.target.value)}
            />
            <button
              onClick={() => {
                renameLabel(label, newLabel);
                setShowRename(false);
              }}
              className="btn btn p-0"
            >
              <FaCheck size={20} className="m-2" />
            </button>
          </div>
        </td>
      )}
    </tr>
  );
};

/**
 * Component to manage the labels of a project
 */

export const LabelsManagement: FC<LabelsManagementProps> = ({
  projectSlug,
  currentScheme,
  availableLabels,
  kindScheme,
  reFetchCurrentProject,
}) => {
  const { notify } = useNotifications();

  // hooks to manage labels
  const { addLabel } = useAddLabel(projectSlug || null, currentScheme || null);
  const { deleteLabel } = useDeleteLabel(projectSlug || null, currentScheme || null);
  const { renameLabel } = useRenameLabel(projectSlug || null, currentScheme || null);

  // manage label creation
  const [createLabelValue, setCreateLabelValue] = useState('');
  const handleCreateLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.value.includes('|')) {
      notify({ type: 'error', message: 'Label name cannot contain |' });
      return;
    }
    setCreateLabelValue(event.target.value);
  };
  const createLabel = () => {
    addLabel(createLabelValue);
    setCreateLabelValue('');
    reFetchCurrentProject();
  };

  return (
    <div>
      <span className="explanations">
        {' '}
        The current scheme is a <b>{kindScheme}</b>
      </span>

      <div className="rounded-2xl bg-white">
        <table>
          <thead className="text-xs text-gray-600 uppercase bg-gray-100">
            <tr>
              <th scope="col" className="px-4 py-3">
                Label
              </th>
              <th scope="col" className="px-4 py-3 text-center">
                Count
              </th>
              <th scope="col" className="px-4 py-3 text-center"></th>
              <th scope="col" className="px-4 py-3 text-center"></th>
            </tr>
          </thead>
          <tbody>
            {availableLabels.map((label, _) => (
              <LabelCard
                key={label}
                label={label}
                removeLabel={() => deleteLabel(label)}
                renameLabel={renameLabel}
              />
            ))}
            <hr className="table-delimiter" />
            <tr>
              <td>
                <input
                  type="text"
                  id="new-label"
                  value={createLabelValue}
                  onChange={handleCreateLabelChange}
                  placeholder="Enter new label"
                  className="form-control"
                />
              </td>
              <td>
                <button onClick={createLabel} className="btn btn p-0">
                  <FaPlusCircle size={20} className="m-2" />
                </button>
              </td>
              <td></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
